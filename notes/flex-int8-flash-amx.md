# 路线 C：整数 QK flash + AMX int8 GEMM

> 2026-09-03。改动在 `vendor/burn-int8-flash-amx`（`5abc55e`）。
> 对拍数字见 `notes/poc-results.md`「整数 flash + AMX」。
> 不叠在 #6 / #9 上：这是新的 flex kernel，codegen 仍走 `module::attention`。

#9 融 GELU/LN 之后，512 `forward_raw` 仍约 **636 ms / 11.8×** Rust ort 53.8 ms。
隔离：flash 205 ms（32%）、72× MMI 228 ms（36%）。这一刀同时打这两块。

## 做了什么

### 1. 整数 QK（路线 C-lite）

Burn 的 `attention()` 仍是 f32 API。`seq_q, seq_kv ≥ 256` 且有 AVX512-VNNI 时：

1. 每个 head 对 Q、K 做 ONNX 式 DQL（`u8`）
2. 一次 VNNI `u8×u8 → i32` 算 `Q @ K^T`（K 转成 `[D,S]`）
3. 反量化 × `1/√d`、mask / bias、行 softmax（f32）
4. `P @ V` 仍是 f32（`val_dim==32` 走 AVX-512 FMA）

不物化 `[H,S,S]`（一次一个 head）。V 不量化，避免再漂一档。
有 softcap 或短序列 → 仍走原来的 f32 flash，现有单测 bit-exact。

**不是**完整的「QK+softmax+PV 全整数 streaming」。那要新 API / 新 codegen。
这一刀只把 QK 从 ~12 GOPS 的小 K f32 gemm 换成已经在跑的 VNNI。

### 2. AMX `tdpbusd`

Rust 的 AMX intrinsic 还不稳定（`x86_amx_intrinsics`）。用 stable `asm!`：
`ldtilecfg` / `tileloadd` / `tdpbusd` / `tilestored` / `tilerelease`。

- 对齐形状：`M%16==0 && N%16==0 && K%64==0`（e5 的 QKV / FFN 都齐）
- 先 `arch_prctl(XFEATURE_XTILEDATA)`，再用一发 16×16×64 探针确认
- 不对齐或探针失败 → 回退现有 VNNI
- 有 rayon 的大 `M` 仍按 MC=64 切

`unsafe` 只在 cubek/flex 的 kernel 文件里（和现有 VNNI 一样），没进 burn-e5q / burn-onnx。

## 单测（`cargo test -p burn-flex --release --lib`）

- `int_gemm::*` 10/10，含 `gemm_amx_aligned_u8_i8_matches_naive`（16×16×64 和 64×64×384 + zp）
- `attention*` 17/17，含 `int8_flash_close_to_f32_naive_e5_like`（256×256×32，vs naive cos > 0.98）
- 短序列 / 原 flash 单测仍走 f32 路径

## 预期

隔离估：flash 205 → ~80–120 ms；MMI 228 → 若 AMX 吃满可能腰斩。
整网乐观 ~400 ms 档，仍远于 2×（108 ms）。数字以 `forward_raw` 为准。
