# 拆 1.12 s：融合 attn 之后还剩什么

> 2026-09-03。同一台 4 核 Xeon（`avx512_vnni` + `amx_int8`）。
> 栈：融合 int8 SDPA → `attention()`（#6）+ flash 按 head 并行（#7）。
> 命令：`cargo run --release -p e5-embed --bin breakdown`
> Rust ort 基线：arena off，`session.run` **53.8 ms / 512 tok**（预编码 ids，不含 tokenizer）。

这一刀**不改模型、不写 kernel**。用同后端、同形状的隔离计时，把 1.12 s 对到具体算子上。

## TL;DR

先前把 512 的 **1.12 s / 21×** 当成模型时间，**口径错了**。

| 口径 | burn | Rust ort | 倍数 |
|---|---:|---:|---:|
| `embed_passages`（sentencepiece + 模型） | **1099 ms** | — | **20×**（不公平） |
| `forward_raw`（只有模型） | **639 ms** | **53.8 ms** | **12×** |
| 其中 sentencepiece（2915 字 → 512 id） | **457 ms** | 0（预编码） | — |

模型本体 **639 ms 已被隔离块加总解释（差 1%）**。没有大块「生成图税」。
`compare_ort` / `mem_stress` 的 512 行用的是 `gen_ref.py` 那条 `* 55` 长串，先整段 encode 再截 512。

模型内三块几乎均分剩下的 12×：

| 块 | 512 ms | 占模型 |
|---|---:|---:|
| **72× MMI（VNNI u8×i8）** | **228** | **36%** |
| **12× f32 flash attention** | **205** | **32%** |
| **12× 展开 GELU / `erf`** | **117** | **18%** |
| 展开 LN ×25 | 44 | 7% |
| DQL ×48 | 33 | 5% |
| 反量化 / reshape / embed | 7 | 1% |
| **隔离之和** | **635** | **99%** |

16 tok：隔离 21 ms，真实 `embed_passages` 28 ms。短句几乎全是模型，tokenizer 只有几毫秒。

## 图里还剩什么（生成代码计数）

| 算子 | 次数 | 512 形状 |
|---|---:|---|
| `module::attention` | 12 | QKV `[1,12,512,32]` |
| `matmul_integer` | 72 | QKV+out `[512,384]×[384,384]` ×36；FFN1 ×12；FFN2 ×12 |
| `dynamic_quantize_linear` | 48 | 36× `[1,512,384]` + 12× `[1,512,1536]` |
| 展开 LayerNorm | ~25 | `[1,512,384]` |
| 展开 GELU（`erf`） | 12 | `[1,512,1536]` |
| `clone` / `unsqueeze` | 884 / 701 | 胶水（计时上可忽略） |

## 方法

`crates/e5-embed/src/bin/breakdown.rs`：隔离块 → 链式 12 层骨架 → `forward_raw` / `embed_passages` 校准。
dummy 权重显式 U8×I8，走线上同一条 VNNI。flex eager，`black_box` 防 DCE。

## 测数（本机，min of repeats）

### 512 tok 隔离

| 块 | min ms | per-call | GOPS（MAC/s） |
|---|---:|---:|---:|
| flash ×12 | 204.7 | 17.1 | 11.8 |
| MMI QKV+out ×36 | 61.1 | 1.70 | 44.5 |
| MMI FFN1 ×12 | 83.8 | 6.98 | 43.3 |
| MMI FFN2 ×12 | 83.4 | 6.95 | 43.5 |
| DQL ×48 | 33.4 | 0.70 | — |
| 展开 LN ×25 | 44.3 | 1.77 | — |
| 展开 GELU ×12 | 117.2 | 9.76 | — |
| 链式 12 层 / 每层换权重 | 634 / 642 | — | — |
| `forward_raw` | **639** | — | — |
| `embed_passages` | **1099** | — | — |

### sentencepiece 随字数（同一条长串的前缀）

| 字 | ids | ms |
|---:|---:|---:|
| 64 | 22 | 11.1 |
| 200 | 61 | 33.0 |
| 400 | 122 | 64.6 |
| 800 | 245 | 126.9 |
| 1600 | 486 | 253.1 |
| 2915 | 512 | 459.9 |

近似 **0.16 ms/字，线性**。先整段 encode 再 `take(510)`，3k 字就要 460 ms。
`sentencepiece-rs` 这条路径比 HF `tokenizers` / 正常 SP 慢两个数量级。AGENTS.md 禁止重做 tokenizer，这里只记账。

## 被证伪 / 被修正的判断

1. **「1.12 s 主要是生成图税」——否。** `forward_raw − 隔离之和 = 4 ms（1%）`。884 clone、`.val()`、Shape/Gather 不是 512 的大头。
2. **「剩下的时间不在 flash」——半对。** flash 再调 TILE 没用（#7），但它仍占**模型**的 32%。零掉 flash 也只从 639 → ~430 ms（仍 8× ORT）。
3. **「21× vs Rust ort」——计量不一致。** 1.12 s 含 SP；54 ms 不含。公平模型倍数是 **12×**。
4. **MMI 已经「够快、不是瓶颈」——否。** 72 个 VNNI GEMM 合计 228 ms，是模型第一名，单独就有 ORT 整网的 **4.2×**。44 GOPS，离峰值还远。

## 下一刀（按证据排序）

只谈**模型** 639 → ~108 ms（2× ORT）。tokenizer 另案。

1. **先收 FFN 侧面：GELU + LN**（#9，`notes/flex-gelu-ln.md`）——codegen 做成，整网几乎没掉。  
   隔离最多 −56 ms；`forward_raw` 仍 ~636 ms。不要再在这条线上磨 erf / LN。
2. **路线 C：int8 flash**  
   205 ms f32 flash @ 12 GOPS。按现在 VNNI 的 44 GOPS 估 QK ≈ 25–40 ms 量级。能啃掉模型一块，**单独不够**到 2×。
3. **MMI 本身再快**（打包、AMX、QKV 合并）  
   228 ms @ 44 GOPS。不提速 GEMM，flash 做完也还剩 ~200 ms 量级整数乘。
4. **整层 / FFN 融合（MMI + GELU + LN + DQL）**  
   这四块合计仍是大头。ORT 赢在 epilogue 融进 GEMM。
5. **计量**  
   `compare_ort` 的 512 行应拆 tokenize / `forward_raw`。#8 已加。不要再用 1.12 s 当模型基线。

**不要**再调 flash TILE / gemm 并行。**不要**再融单个 DQL。**不要**在 #6 上叠 C。

## 复现

```bash
cargo run --release -p e5-embed --bin breakdown
```
