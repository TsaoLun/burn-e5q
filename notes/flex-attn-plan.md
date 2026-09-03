# 规划：e5 int8 SDPA → Burn `attention()`，512 走 flash

> 操作手册。测数见 `notes/poc-results.md`「融合 attention」；落地记录见
> `notes/flex-attn.md`。本文件只回答 **怎么做、按什么顺序、什么算出做成**。

这句话是 **两件独立的事**，必须拆开排期，不能合成「写一个快的 attention kernel」。

| # | 事 | 仓库 | 目的 |
|---|---|---|---|
| 1 | 把 e5 分解开的 int8 SDPA **收成一个** `Attention` 节点 | `onnx-ir`（TsaoLun/burn-onnx 或本仓 `vendor/*`） | 生成代码改调 `burn::tensor::module::attention` |
| 2 | 让 flex 对 **512×512** 走 flash，而不是 naive | `burn-flex` | 不物化 `[H,S,S]` f32 scores |

1 不做，2 永远碰不到 e5（图里没有 `Attention`）。2 不做，1 在 512 上仍会 `<= 256K` 走 naive，判断无法被检验。

---

## 0. 先看清图，再谈融合

e5 的 ONNX **没有** `Attention`，也 **没有** f32 `MatMul`。opset 11。每层是：

```
Q/K/V proj (MMI + dequant) → reshape [B,S,12,32]
Q: permute [0,2,1,3] → [B,H,S,D]     D=32, H=12
K: permute [0,2,3,1] → [B,H,D,S]     已是 K^T
V: permute [0,2,1,3] → [B,H,S,D]
DQL(Q) DQL(K) DQL(V)
QK: MMI → Cast → Mul(q_s*k_s) → Div(√32) → Add(mask) → Softmax(axis=3)
DQL(softmax)                         // 最大的 DQL：[H,S,S] f32
PV: MMI → Cast → Mul → permute [0,2,1,3] → reshape [B,S,384]
```

约束（写 matcher 时一条都不能漏）：

- Softmax 是不变锚点。axis=3，rank 4，即 last dim。
- mask 是图级 `/Mul_output_0`，shape `[B,1,1,S]` f32，**12 层共享**。
  `is_single_use` 只约束 Add/Softmax 链，**不要**要求 mask single-use。
- `Div` 的除数是 √32（initializer，常为 rank-1 `[5.656854]`）。
  `Attention.scale` 缺省就是 `1/sqrt(head_dim)`；折进去或省略属性，数值应相同。
- 运行时 `Mul(q_s*k_s)` / `Mul(attn_s*v_s)` 是整数路径的反量化，
  **禁止**折进 `Attention.scale`。
- K 相对 Q 的 perm 是末两维对调。Burn `attention()` 要 `[B,H,S,D]`，
  必须补 `Transpose [0,1,3,2]`（已有 f32 pre-scaled 模式同款）。
- 替换点是 PV 的 `Cast → Mul(dequant)` 那个 **Mul**：输出名不能变，
  后面的 Transpose 还挂在这个名字上。
- 简化在 **PHASE 4b**（type inference 之后）。插入的 `Attention` 走
  `build_node` / `extract_config`，**不会**再跑 `infer_types` 的 opset 23
  检查。opset 11 图可以插入。不要为这改 `AttentionProcessor::infer_types`。
- 已有 `coalesce_attention` 只认 **f32 MatMul → Softmax → MatMul**。
  e5 对不上，必须加 int8 分支，不要另开一个 pass。

现成可复用、不要重写：

- `onnx-ir` `coalesce_attention` + PHASE 4b
- `burn-onnx` Attention codegen → `burn::tensor::module::attention`
- flex `attention.rs`：`seq_q*seq_kv < 256K` naive，否则 flash
  （`TILE_KV=64`，online softmax）
- e5 tokenizer / mean-pool / L2 / DQL NodeProcessor：**不要动**

---

## 1. 三条实现路线（先选再写）

收成 `attention()` 只决定 **调用点**。节点里面算什么，有三条路。

### 路线 A — f32 重写（最小切口，已按此落地）

输入用 DQL **之前** 的 f32 Q/K/V，丢掉二次量化 + int8 QK/PV。

```
f32 Q,K,V ──► Attention(scale=1/√d, bias=mask) ──► f32 Y
```

- 不写新 kernel。512 走现成 flex flash。
- 数值相对 ORT 的 int8 QK 会漂，预期仍 0.99x。
- **代价**：丢掉已经很快的 VNNI QK/PV，换成 tiled f32 gemm。
- 测过：512 **1.27 s → 1.15 s（10%）**。图对、内存对、延迟判断被证伪。

只用来回答「能不能接到 `attention()` + flash」。**不要**指望它追上 ORT。

### 路线 B — 只融 Softmax 周围，QK/PV 仍 int8

保留 `MMI(QK)` / `MMI(PV)`，只把 `Softmax + DQL(softmax)` 收短，或
softmax 输出不再物化成完整 `[H,S,S]` u8。

- 保住 VNNI QK。
- Softmax 仍要看见 scores；若不做 streaming，`[H,S,S]` i32/f32 还在。
- 单独做的话，提升量级会接近融 DQL 的 13%，不是分水岭。

### 路线 C — int8 flash（要新 kernel）

一个 kernel 内：u8/i8 QK tile（VNNI）→ 反量化 + scale + mask →
online softmax → u8/i8 PV tile。不物化 `[H,S,S]`，也不退回 f32 GEMM。

- 这才同时满足「收成一个执行单元」和「512 不物化 + 整数算力」。
- Burn 的 `attention()` 今天是 f32 API。C 要么：
  - 扩 `FloatTensorOps::attention` 不够，要新 `int_attention` / 后端私有入口；要么
  - codegen 不走 `module::attention`，走专用 `matmul_integer_flash` 一类。
- **不要**先做 C 来「证明判断」。判断用 A 已经证完（证伪）。
  C 是「还要在 attention 这条线上追 ORT」时的下一刀。

**本规划执行的是 A + 512 flash 门槛。** C 单列，不和 A 绑在同一 PR。

---

## 2. 工作包（按 A 执行）

### WP1 — `coalesce_attention` 的 int8 匹配

文件：`onnx-ir/src/simplify/coalesce_attention.rs`

从 Softmax 往回：

1. 可选 `Add(mask)`：一侧能追溯到 `MatMulInteger`，另一侧是 mask。
2. 可选 `Div/Mul(const)` → `scale = 1/const` 或 `const`。
   抽不出常量时：**仍然剥 Div**，`scale` 留 `None`（默认 `1/sqrt(d)`）。
   禁止 `scale_value.or(Some(1.0))`——e5 第一轮 simplify 时 √32 可能
   还没挂上 `value()`，写成 1.0 会把分数放大 5.66×。
3. 剥运行时反量化 `Mul` → `Cast` → `MatMulInteger`。
4. Q/K = 该 MMI 两个输入各自的 **DQL.inputs[0]**（f32）。
5. Softmax 唯一消费者必须是 DQL；DQL 的 u8 唯一消费者必须是 PV MMI。
6. V = PV MMI 第二输入的 DQL f32。
7. PV MMI 之后 `Cast → Mul`，这个 Mul 换成 `Attention`。

`try_match_sdpa` 里：**int8 分支在 f32 MatMul 路径之前**。匹配失败再走原逻辑。
原有 f32 SDPA 单测一条都不能坏。

单测至少 4 条（没有这些不要宣称 matcher 完成）：

- e5 布局（Q/K/V transpose + 3×DQL + MMI + Cast + Mul + Div + Add + Softmax + DQL + MMI + Cast + Mul）
- mask 另有 Relu 消费者仍匹配
- Div 除数是 dynamic scalar 仍匹配，且 **不**写 `scale` 属性
- Softmax 后面没有 DQL → 不匹配

### WP2 — K 布局

Q perm `[0,2,1,3]` vs K perm `[0,2,3,1]` = `is_perm_with_last_two_swapped`。
在 **QK MMI 的 node slot** 上写 corrective `Transpose [0,1,3,2]`
（与现有 f32 pre-scaled 模式相同），Attention 的 K 用纠正后的名字。

看不到 transpose 就不要猜。猜错会静默算错 attention。

### WP3 — codegen / 语义，不要改 processor

- mask：rank-4 f32 → codegen 已有的 `attn_bias` 路径。不要改成 bool mask。
- `Attention.scale`：抽到 Div 就写成 `1/√32`；抽不到就省略。
- 输出名 = 被替换的 Mul 的输出名。
- **不要**改 `AttentionProcessor::infer_types` 的 opset 23。
- **不要**在 burn-onnx 加 `unsafe`。
- **不要**重做 DQL NodeProcessor / tokenizer / e5 pipeline。

验收（编译期，不跑模型也能量）：

```
grep module::attention  generated.rs   # = 12
grep activation::softmax generated.rs  # = 0
grep dynamic_quantize_linear           # 96 → 48
grep matmul_integer                    # 96 → 72
```

少一层就回去查 matcher，不要先跑 `compare_ort`。

### WP4 — flex：512 必须走 flash

文件：`burn-flex/src/ops/attention.rs`

```
NAIVE_SCORE_BUDGET = 256 * 1024   // 512*512 = 256K 刚好踩线
```

旧条件是 `seq_q * seq_kv <= budget` → 512 **走 naive**，物化 `[H,S,S]`，
WP1 的「不物化」判断无法成立。

改成 **严格小于**：

```
if seq_q * seq_kv < NAIVE_SCORE_BUDGET { naive } else { flash }
```

16×16 仍 naive（调度 + 小矩阵，flash 更亏）。不要用 `seq_kv > 8*TILE_KV`
那条过时注释当依据，以 score 元素数为准。

这是 **一行 + 注释** 的 burn vendor 变更，和 WP1 可以同一个 PoC PR，
但 commit 必须分开（onnx-ir vs flex）。

### WP5 — pin、对拍、怎么读数

1. 推 `vendor/burn-onnx-coalesce-int8-attn`、`vendor/burn-flash-512`（或等价名）。
   本仓 `Cargo.toml` 两处 `rev` 一起改。`cursor[bot]` 不能推 TsaoLun fork。
2. **测前** commit / push / PR。
3. `cargo run --release -p e5-embed --bin compare_ort`
4. `cargo run --release -p e5-embed --bin mem_stress -- 5 2048`
5. 对照用本机 Rust ort（`ort-mem`，arena off），不要用 `ref_data.json` 里的 Mac Python 数。

读数（用来证明或证伪「瓶颈在 `[H,S,S]`」）：

| 信号 | 判断成立 | 判断不成立 |
|---|---|---|
| 512 延迟 | 1.27 s → **300–500 ms**（相对 54 ms ORT 进个位数倍） | 几乎不动（~10%） |
| 4×512 HWM | 相对 315 MB 明显下降 | 不降 = 仍在物化，先查 WP4 |
| 短句 16 tok | 允许几乎不动 | — |
| mean cos | 仍约 0.99x；top-1 不崩 | <0.98 或检索崩 → 停，改路线 B/C |

实测（2026-09-03）：延迟列走「不成立」，HWM 315→278，cos 0.9946，top-3 2/2。
**WP1–WP4 作为接线任务算完成；作为性能杠杆算证伪。**

---

## 3. 不要做

- 不要写 streaming int8 kernel 来完成这句话（那是路线 C）。
- 不要先做 AMX / 整层 QKV+FFN 融合（另一条线）。
- 不要把 runtime `q_s*k_s` 折进 scale。
- 不要要求 mask single-use。
- 不要在 burn-e5q / burn-onnx 加 `unsafe`。
- 不要 commit 118 MB ONNX / `.bpk`。
- 不要对 tracel-ai 开 PR，不要 `git push --force` 到 `main`。
- 不要重做 DQL NodeProcessor、tokenizer、`graph.rs` boundary scalar、FP32 model-check。
- 不要把 `ort` 链进 `e5-embed`（污染 RSS）。

---

## 4. 顺序（一次只做一刀）

```
WP1 matcher + 单测
    → WP2 K 布局（可与 WP1 同一 commit）
    → cargo test -p onnx-ir coalesce
    → vendor push burn-onnx
    → WP4 flex `< 256K`（单独 commit / 单独 vendor）
    → 本仓 pin + 测前 PR
    → 看生成代码 12 / 0 / 48 / 72
    → compare_ort + mem_stress
    → 按 §2 表格读数，写 poc-results
```

A 证伪之后 **停**。1.12 s 的拆解见 `notes/gap-breakdown.md`：
`embed_passages` 含 457 ms sentencepiece；模型是 **639 ms / 12×** ORT，
已被 MMI 36% + flash 32% + GELU 18% 解释。下一刀另开规划，按证据是：

1. 整层 / FFN 融合（MMI + GELU + LN + DQL，66% 模型）
2. 路线 C：int8 flash（32% 模型，单独不够到 2×）
3. MMI 本身再快（44 GOPS，228 ms 已是 ORT 整网的 4×）

不要在 A 的 PR 上继续叠 C。不要再给 f32 flash 换 gemm（#7 已证伪）。
