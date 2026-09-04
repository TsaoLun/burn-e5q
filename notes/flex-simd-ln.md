# AVX-512 last-axis LayerNorm（D=384）

> 2026-09-04。改动在 `vendor/burn-simd-ln`（`fbe1288`），叠在 AMX packed-B 上。
> 对拍数字见 `notes/poc-results.md`「AVX-512 LN」。
> 不改 TILE。不挂钩 C-lite。不是再融单个 DQL codegen。不是整层 FFN 融合。

AMX pack 之后 512 `forward_raw` **123.6 ms / 2.3×** 本机 Rust ort 53.5 ms。
隔离 fused LN ×25 仍是 **21.8 ms**。e5 的 25 个 LN 全是最后一维 D=384。

## 做了什么

1. **AVX-512 行 kernel**（`D % 16 == 0`）  
   偏方差与 macerator 路径相同：`var = E[x²] − E[x]²`。按 **4 行**一块做
   affine，γ/β 只 load 一次。

2. **unique 入口原地写**  
   `to_contiguous()` 对已打包的 buffer 会 `clone` 掉 uniqueness。改成
   已 contiguous 就拿走所有权。unique 时两趟扫同一块，不再 `vec![0; n]`。

3. 其它宽度仍走 macerator。C-lite 仍不挂钩。TILE 仍是 64。

`unsafe` 只在 burn-flex。

## 单测

`cargo test -p burn-flex --release --lib -- layer_norm ln_`：18/18。

- 原有 LN / rayon / tail
- 新增 D=384 vs 参考（2e-5）
- 原地 == copy
- 隔离 `[512,384]`：AVX-512 **0.055 ms** vs 标量 **0.126 ms**；×25 ≈ 1.4 vs 3.2

## 对拍（本机 4 核 Xeon，flex release；进程分开跑）

待 `compare_ort` / `breakdown` / `mem_stress` / `ort-mem` 填。
对外 512 用 compare_ort 的 `forward_raw`。不要用 Mac Python 4.3 / 201。
