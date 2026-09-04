# D=32 AVX-512 flash + 不再展开 `[1,1,1,S]`

> 2026-09-04。改动在 `vendor/burn-flash-d32`（`a69b3a5`），叠在 AMX 上。
> 对拍数字见 `notes/poc-results.md`「D=32 flash」。
> 不挂钩 C-lite。不改 TILE。

AMX 之后 512 `forward_raw` **414 ms / 7.7×** 本机 Rust ort 53.8 ms。
隔离第一名是 **f32 flash 208 ms（50%）**。C-lite（VNNI QK + 物化 `[S,S]`）已经更慢。

## 做了什么

### 1. 长序列 D=32 走专项 AVX-512 flash

`seq ≥ 64`、`head_dim == val_dim == 32`、f32、无 softcap、有 `avx512f` 时：

1. 每个 KV tile（64）把 K 转成 `[32, 64]`，沿 KV 轴 FMA 算 QK，scale 融进去
2. 满 64 宽的 tile 用 Cephes `exp` + `_mm512_scalef_ps` 做 online softmax
3. PV 用两路 zmm 累加（和 `attention_int8` 的 D=32 一样）

算法仍是 flash（不物化 `[S,S]`）。短序列 / 别的 D / softcap 仍走 `gemm::gemm`。

### 2. `[1,1,1,S]` mask/bias 不再展开成 `[H,S,S]`

e5 的 attention bias 是 `[B,1,1,S]`。旧 helper 因为 `seq_q` 是 1 就
`expand` 成 `[1,12,512,512]`（每层 12 MiB）。现在 `seq_kv` 对齐时只留一行，
`q_step = 0`。只在 `seq_kv` 自己要广播时才 expand。

## 单测（`cargo test -p burn-flex --release --lib attention`）

23/23。含：

- `d32_flash_close_to_naive_and_gemm_flash`（128，cos > 0.999）
- `d32_flash_partial_tile_and_bias`（100 + `[1,1,1,S]`）
- `d32_exp_tracks_libm`（相对误差 < 2e-5）
- `test_flash_bias_broadcast_seq_q`（`[1,1,1,S]` vs 展开）
- 隔离 12×512：d32 **2.4 ms/层** vs gemm-flash **3.6 ms/层**（×12 ≈ 29 vs 43）

## 对拍

见 `notes/poc-results.md`。读数用 `forward_raw` vs 本机 Rust ort 3.5 / 1099 / 53.8。

## 不要做

不要再挂钩 C-lite。不要再调 TILE。不要用 A&S 换默认 erf。
