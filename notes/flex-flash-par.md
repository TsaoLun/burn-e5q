# flex flash 按 head 并行

> 2026-09-03。A（int8 SDPA → `attention()` + 512 flash）只快了 10%。
> flash 路径对 12 个 head **串行**，gemm 还是 `Parallelism::None`。
> 这一刀验证「1.15 s 是不是卡在 flash」。

## 做了什么

`vendor/burn-flash-par-heads` `fd4f793`：`attention_flash` 用
`par_chunks_mut` 按 head 切开。每个 head 写自己的输出切片，没有新的
`unsafe`。gemm 保持串行，避免嵌套 rayon。

单测：原有 14 条 + `test_flash_12_heads_matches_naive`（e5 的 12 head）。

## 对拍（本机 4 核 Xeon，flex）

| 场景 | 融合 attn（串行 head） | **head 并行** | vs 串行 |
|---|---:|---:|---:|
| 16 tok | 29.5 ms | **28.4 ms** | 1.04× |
| packed batch | 5.57 s | **5.44 s** | 1.02× |
| 512 tok | 1.15 s | **1.12 s** | 1.03× |

mean cos **0.9946**（与串行 flash 逐 case 相同）。top-3 **2/2**。
`mem_stress -- 5 2048`：稳态 **213 / 278 MB**（与串行相同）；4×512 ~4.6 s（串行 ~4.9 s）。

## 判断

512 只动了 3%，在噪声里。**剩下的 ~1.12 s 主要不在 flash。**
head 并行是对的（数值没漂），但不是杠杆。不要再调 TILE / gemm 并行来追 ORT。

下一刀：整层执行单元（72 个 MMI + LN 的调度），或路线 C（int8 flash，把 QK 拉回 VNNI）。再啃 f32 flash 只会再动几个百分点。
