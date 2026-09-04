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
上一刀 codegen 融 GELU 时隔离能省、整网几乎没动（图路径不 unique）。
这次 in-place 和 alloc 两条都走 SIMD，整网吃到了。

## 单测（`cargo test -p burn-flex --release --lib gelu`）

6/6。含：

- 原有 `test_gelu_small_matches_libm` / `test_gelu_parallel_matches_libm`（1e-6）
- `erf_avx512_tracks_libm_dense`：16025 点，max abs **6.0e-8**
- `gelu_e5_ffn_shape_matches_libm`：`[512,1536]`
- 隔离 `[512,1536]`：AVX-512 **0.77 ms** vs 标量 libm **3.96 ms**（×12 ≈ 9 vs 48）

## 对拍（本机 4 核 Xeon，flex release）

| 口径 | D=32 flash | **SIMD GELU** | 本机 Rust ort | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok `forward_raw` | 13.9 | **10.9** | 3.5 | **3.1×** |
| packed batch `embed_passages` | 2928 | **2401** | 1099 | **2.2×** |
| 512 `forward_raw` | 350 | **269** | 53.8 | **5.0×** |
| 512 `embed_passages` | 804 | **731** | 53.8 | 含 ~451 ms SP |

mean cos **0.9950**（min **0.9876**），ranking 2/2。
隔离 fused GELU ×12：**83 → 17 ms**。`forward_raw` **350 → 269**（−81 ms）。
`mem_stress -- 5 2048`：4×512 **3050 ms**，RSS **213 / 232 MB**。

512 还差 ~161 ms 才到 2×（~108 ms）。大头变成 D=32 flash（隔离 132 / ~50%）
和整层 FFN 融合（MMI + GELU + DQL + LN 少扫几趟 `[1,512,1536]`）。
不要再调 TILE / 再挂钩 C-lite / 再用 A&S 换默认 erf。
