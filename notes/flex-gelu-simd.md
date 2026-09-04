# AVX-512 GELU（musl/fdlibm `erff`）

> 2026-09-04。改动在 `vendor/burn-simd-gelu`（`a62f534`），叠在 D=32 flash 上。
> 对拍数字见 `notes/poc-results.md`「AVX-512 GELU」。
> 不换 A&S。不改 unary `erf`（仍是 `libm::erff`）。

D=32 flash 之后 512 `forward_raw` **350 ms / 6.5×** 本机 Rust ort 53.8 ms。
隔离里 fused GELU 的标量 `erff` 仍约占 **80 ms**。

## 做了什么

连续 f32 GELU（含 e5 FFN `[1,512,1536]`）走 AVX-512：

1. 系数和分段与 musl / libm `erff` 相同（SunPro fdlibm `s_erff.c`）
   - `|x| < 0.84375`：`x + x·P(x²)/Q(x²)`
   - `0.84375 ≤ |x| < 1.25`：`ERX + P(s)/Q(s)`，`s = |x|-1`
   - `1.25 ≤ |x| < 6`：`1 - erfc`，chop + 两路 Cephes `exp`
   - `|x| ≥ 6`：饱和到 `±1`
2. 大 buffer 仍按 16K `par_chunks`；每个 worker 开 FTZ+DAZ
3. 非连续 / 非 f32 仍走原来的 `unary_op` + `libm`

不是 Abramowitz–Stegun 7.1.26。`Tensor::erf` 默认路径没动。

## 单测（`cargo test -p burn-flex --release --lib gelu`）

6/6。含：

- 原有 `test_gelu_small_matches_libm` / `test_gelu_parallel_matches_libm`（1e-6）
- `erf_avx512_tracks_libm_dense`：16025 点，max abs **6.0e-8**
- `gelu_e5_ffn_shape_matches_libm`：`[512,1536]`
- 隔离 `[512,1536]`：AVX-512 **0.77 ms** vs 标量 libm **3.96 ms**（×12 ≈ 9 vs 48）

## 对拍（本机 4 核 Xeon，flex release）

测前 revision。`compare_ort` / `breakdown` / `mem_stress` 数字随后补。

不要再用 A&S 换默认 erf。不要再调 flash TILE / 再挂钩 C-lite。
