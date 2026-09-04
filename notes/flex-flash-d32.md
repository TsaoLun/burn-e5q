# D=32 AVX-512 flash + 不再展开 `[1,1,1,S]`

> 2026-09-04。改动在 `vendor/burn-flash-d32`（`219fe61`），叠在 AMX 上。
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

### 3. packed batch 的两个坑（都踩过，都修了）

- **96-way rayon**：`[8,12,512]` 一 head 一任务，4 核 L2 被踩烂。按
  `num_cpus` 切块，B=8 隔离 flash 回到 **8.1×** B=1。
- **denormal**：compare_ort 把 6 条短句和 1 条 512 打成 `[7,512]`，pad
  位在 QK/PV 里出 denormal。没开 MXCSR FTZ+DAZ 时 **~19 s**；
  `mem_stress` 全 512 实 token 的 `[8,512]` 只有 ~2 s 模型时间。每个
  worker 用 `stmxcsr` / `ldmxcsr` 打开 FTZ+DAZ。

## 单测（`cargo test -p burn-flex --release --lib attention`）

23/23。含：

- `d32_flash_close_to_naive_and_gemm_flash`（128，cos > 0.999）
- `d32_flash_partial_tile_and_bias`（100 + `[1,1,1,S]`）
- `d32_exp_tracks_libm`（相对误差 < 2e-5）
- `test_flash_bias_broadcast_seq_q`（`[1,1,1,S]` vs 展开）
- 隔离 12×512：d32 **2.4 ms/层** vs gemm-flash **3.6 ms/层**（×12 ≈ 29 vs 43）
- 隔离 8×12×512：d32 **19.6 ms（8.1×）**

## 对拍（本机 4 核 Xeon，flex release）

| 口径 | AMX + gemm flash | **D=32 flash** | 本机 Rust ort | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok `forward_raw` | 13.2 | **13.9** | 3.5 | **4.0×** |
| packed 7 条 `embed_passages` | 3829 | **2928** | 1099 | **2.7×** |
| 512 `forward_raw` | 414 | **350** | 53.8 | **6.5×** |
| 隔离 flash ×12 | 208 | **129** | — | −79 ms |

mean cos **0.9950**（min 0.9876），ranking 2/2。
`mem_stress -- 5 2048`：4×512 **3511 ms**，RSS **213 / 232 MB**。

512 还差 ~240 ms 才到 2×（~108 ms）。下一刀仍是 GELU `erff`（隔离 ~82）
和整层融合，不是再调 TILE / 再挂钩 C-lite。

## 不要做

不要再挂钩 C-lite。不要再调 TILE。不要用 A&S 换默认 erf。
不要在 4 核上对 96 个 head 各开一个 AVX-512 任务。
不要忘了 FTZ：只拿全 512 实 token 的 `mem_stress` 当 packed 会漏掉 denormal。
