# flex flash 按 head 并行

> 2026-09-03。A（int8 SDPA → `attention()` + 512 flash）只快了 10%。
> flash 路径对 12 个 head **串行**，gemm 还是 `Parallelism::None`。
> 这一刀验证「1.15 s 是不是卡在 flash」。

## 做了什么

`vendor/burn-flash-par-heads` `fd4f793`：`attention_flash` 用
`par_chunks_mut` 按 head 切开。每个 head 写自己的输出切片，没有新的
`unsafe`。gemm 保持串行，避免嵌套 rayon。

单测：原有 14 条 + `test_flash_12_heads_matches_naive`（e5 的 12 head）。

## 读数

| 信号 | 成立 | 不成立 |
|---|---|---|
| 512 延迟 | 相对 1.15 s 掉一截（接近 4 核，理想 ~2–3× flash 部分） | 几乎不动 → 1.15 s 主要不在 flash |
| 短句 | 允许不动（仍走 naive） | — |
| mean cos | 与融合 attn 同量级 | 数值崩了先停 |

对拍数字补在测后。
