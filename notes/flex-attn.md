# 融合 int8 SDPA → Burn f32 attention

> 2026-09-03。要证明的判断：VNNI + 融 DQL 之后，512 tok 仍剩 ~1.2 s
> **既不是 GEMM 也不是 DQL**；ORT 赢在「一整块 Transformer 当执行单元 +
> 不物化 `[heads,S,S]`」。对拍数字见 `notes/poc-results.md`「融合 attention」。

## 做了什么

e5 的 ONNX **没有** `Attention` 节点，也没有 f32 `MatMul`。每层是：

```
Q/K/V proj (MMI+dequant) → reshape [B,S,12,32]
Q: permute [0,2,1,3] → [B,H,S,D]
K: permute [0,2,3,1] → [B,H,D,S]   // 已是 K^T
V: permute [0,2,1,3] → [B,H,S,D]
DQL(Q), DQL(K), DQL(V)
QK: MMI → Cast → Mul(q_s*k_s) → Div(√32) → Add(mask) → Softmax
DQL(softmax)   // 最大的 DQL，[12,S,S] f32
PV: MMI → Cast → Mul → permute → reshape [B,S,384]
```

`onnx-ir` 已有的 `coalesce_attention` 只认 f32 `MatMul → Softmax → MatMul`，
对不上。这次在同一 pass 里加了 int8 模式：

1. **`vendor/burn-onnx-coalesce-int8-attn` `f78e156`**
   - 从 Softmax 往回剥 Add(mask) / Div(√d) / 反量化 Mul / Cast，落到
     `MatMulInteger`；Q/K/V 取 DQL **之前** 的 f32。
   - 用已有 `Attention` 节点替换 PV 反量化 Mul（输出名不变）。
   - K 的 `[0,2,3,1]` 相对 Q 的 `[0,2,1,3]` 是末两维对调，复用 corrective
     `Transpose [0,1,3,2]`。
   - 运行时 `q_s*k_s` **不**折进 `Attention.scale`。mask 是图级共享的
     `[B,1,1,S]` 加法 bias，不要求 single-use。
   - 单测 4 条（e5 布局、共享 mask、动态 Div、缺 softmax DQL 不匹配）+
     原有 25 条 f32 SDPA 仍过。

2. **`vendor/burn-flash-512` `245ab35`**
   - flex `attention()` 的 naive 预算从 `<= 256K` 改成 `< 256K`。
   - 512×512 = 256K 刚好压在旧边界上，会物化 `[H,S,S]`；改完后走 flash
     （online softmax，不物化完整 scores）。16 tok 仍走 naive。

没有新 kernel，没有 `unsafe`。数值会相对 ORT 的 int8 QK/PV 路径漂移，
预期仍在 0.99x；若崩再改 streaming int8。

## 对拍（本机 4 核 Xeon，flex）

生成代码：**12** 次 `burn::tensor::module::attention`，**0** 个 Softmax，
DQL 96→48，MatMulInteger 96→72。scale 是 `1/√32`，mask 是图级
`[B,1,1,S]` 加法 bias，K 有 corrective `permute([0,1,3,2])`。

| 场景 | 融 DQL | **融合 attn** | vs 融 DQL | **vs 本机 Rust ort** |
|---|---:|---:|---:|---:|
| 16 tok | 29.6 ms | **29.5 ms** | 1.00× | **12×**（2.4 ms） |
| packed batch | 7.07 s | **5.57 s** | 1.27× | **5.9×**（936 ms） |
| 512 tok | 1.27 s | **1.15 s** | 1.10× | **21×**（53.8 ms） |

mean cos **0.9946**（min 0.9861，融 DQL 是 0.9960 / 0.9935）。
top-3 检索从 0/2 变成 **2/2**。加载 RSS 87.5 MB；`mem_stress -- 5 2048`
稳态 **213 / 278 MB**（HWM 相对融 DQL 的 315 降 37 MB）。

## 判断被证伪（延迟）

预设分水岭是 512 掉到 300–500 ms。实际只掉 **10%**。图已经不再物化
`[12,S,S]`（HWM 也降了），所以剩下的 ~1.15 s **不是** softmax / score
矩阵。更大的账在：

1. 每层还剩 6 个 int8 GEMM（QKV + out + FFN），72 个 MMI 仍走逐 op 调度。
2. 把 int8 QK/PV（VNNI）换成 f32 flash，等于丢掉一条已经很快的整数路径，
   换来的 tiled f32 gemm 并不便宜。
3. LayerNorm / 其余 eager 调度税没动。

短句完全不动，与「16 tok 走 naive、调度主导」一致。

要再追 ORT，下一刀不该是再融一个 DQL，而是 **整层当执行单元**
（QKV+attn+FFN），或给 flash 一条不慢于 VNNI QK 的实现。
