# 路线 C：整数 QK flash（未挂钩）+ AMX int8 GEMM

> 2026-09-03。改动在 `vendor/burn-int8-flash-amx`（`21dba0c`）。
> 对拍数字见 `notes/poc-results.md`「整数 flash + AMX」。
> 不叠在 #6 / #9 上：这是新的 flex kernel，codegen 仍走 `module::attention`。

#9 融 GELU/LN 之后，512 `forward_raw` 仍约 **636 ms / 11.8×** Rust ort 53.8 ms。
隔离：flash 205 ms（32%）、72× MMI 228 ms（36%）。这一刀两块都试了。

## 做了什么

### 1. 整数 QK（路线 C-lite）——写成了，测完没挂钩

`seq_q, seq_kv ≥ 256` 且有 AVX512-VNNI 时可以：

1. 每个 head 对 Q、K 做 ONNX 式 DQL（`u8`）
2. 一次 VNNI `u8×u8 → i32` 算 `Q @ K^T`
3. 反量化 × `1/√d`、mask / bias、行 softmax（f32）
4. `P @ V` 仍是 f32（`val_dim==32` 走 AVX-512 FMA）

**e5 512 实测 ~280 ms，比 tiled f32 flash 的 205–208 ms 更慢。**
DQL + 物化 `[S,S]` + K=32 的 VNNI 摊不赢 `gemm` crate 的 f32 tile。
kernel 留着给单测；`attention_flash` **不**走这条。完整路线 C（QK+PV 都整数、不物化 scores）还没做。

### 2. AMX `tdpbusd`——这是这一刀的赢面

Rust 的 AMX intrinsic 还不稳定（`x86_amx_intrinsics`）。用 stable `asm!`：
`ldtilecfg` / `tileloadd` / `tdpbusd` / `tilestored` / `tilerelease`。

- 对齐形状：`M%16==0 && N%16==0 && K%64==0`（e5 的 QKV / FFN 都齐）
- 先 `arch_prctl(XFEATURE_XTILEDATA)`，再用一发 16×16×64 探针确认
- 不对齐或探针失败 → 回退现有 VNNI
- 有 rayon 的大 `M` 仍按 MC=64 切

`unsafe` 只在 flex kernel 里（和现有 VNNI 一样），没进 burn-e5q / burn-onnx。

## 单测（`cargo test -p burn-flex --release --lib`）

- `int_gemm::*` 10/10，含 `gemm_amx_aligned_u8_i8_matches_naive`（16×16×64 和 64×64×384 + zp）
- `attention*` 17/17，含 `int8_flash_close_to_f32_naive_e5_like`（256×256×32，vs naive cos > 0.98）

## 对拍（本机 4 核 Xeon，flex release）

隔离 512（min of repeats）：

| 块 | 融 GELU/LN 后 | **AMX + f32 flash** | 差 |
|---|---:|---:|---:|
| flash ×12 | 205 | **208** | 噪声（仍是 f32 tile） |
| MMI 72 | 228 @ 44 GOPS | **21 @ 280–600 GOPS** | **−207 ms（~11×）** |
| `forward_raw` | 636 | **417** | **−219 ms** |

整数 QK 挂钩时 flash 279 ms、`forward_raw` 483 ms，所以拔掉了。

| 场景 | 融 GELU/LN | **AMX** | Rust ort | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok | 28.6 ms | **17.0 ms** | 2.4 ms | **7.1×** |
| packed batch | 5.60 s | **3.83 s** | 936 ms | **4.1×** |
| 512 `forward_raw` | 636 ms | **408–417 ms** | 53.8 ms | **7.6–7.8×** |

mean cos **0.9950**（min 0.9886），ranking 2/2。
`mem_stress -- 5 2048`：4×512 稳态 **212 / 275 MB**，约 3.95 s/round。

## 结论

MMI 不再是模型第一名（21 ms / 5%）。现在第一名是 **f32 flash 208 ms（50%）**，然后展开记账的 GELU 117（图上已是 fused ~83）。
到 2×（~108 ms）还差大约 300 ms，大头在 flash 和 GELU 的 `erff`，不是 GEMM。
不要再挂钩这版 C-lite。不要再调 TILE。QKV 合并相对 AMX 已经没必要。
