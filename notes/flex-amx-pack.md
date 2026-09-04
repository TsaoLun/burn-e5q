# AMX packed-B 缓存 + SIMD pack/zp

> 2026-09-04。改动在 `vendor/burn-amx-pack`（`2a05a84`），叠在 Q-block flash 上。
> 对拍数字见 `notes/poc-results.md`「AMX packed-B」。
> 不改 TILE。不挂钩 C-lite。不是再融单个 DQL codegen。

Q-block 之后 512 `forward_raw` **149 ms / 2.8×** 本机 Rust ort 53.5 ms。
隔离里 FFN1 ×12（512×384×1536）是 **27.8 ms @ 130 GOPS**，FFN2 同 MAC
只要 **6.0 ms @ 602 GOPS**。差在每次 forward 标量 `pack_b`（576 KB，stride=N）
和 zp / `sum_cols`。权重指针进程内不变，可以缓存。

## 做了什么

1. **packed-B + 列和缓存**  
   key = `(ptr, n, k, buf_tag)`。只缓存 `B_IS_I8`（e5 权重）。转换过的
   临时 u8 B 不缓存。满 96 条清空。`buf_tag` 挡住同尺寸分配复用指针。

2. **SSE2 `pack_b`**  
   4 行 ×16 N 用 `_mm_unpacklo/hi_epi8` + `epi16` 交织成 AMX tile。

3. **AVX-512 `sum_cols` / `apply_zp`**  
   16 宽 `cvtepi8` + `mullo_epi32` 环绕乘。`da/db` 与 zp 共用同一份 sums。

4. **AMX 串行**  
   `tdpbusd` 加进程锁。并行测试（以及可能重叠的 MMI）会踩 XSTATE，
   曾出现正好一个 16×16 C tile 损坏。缓存命中后 512×384×1536 ~0.8 ms，
   串行够用。

C-lite 仍不挂钩。TILE 仍是 64。

## 单测

`cargo test -p burn-flex --release --lib -- int_gemm attention`：34/34。

- 原有 AMX 对齐形状 + FFN1 短面板 `(32, 1536, 384)` vs naive
- 新增：同一 B 指针 512×384×1536，later 对得上 first；隔离
  first **1.69 ms** / later **0.76 ms**

## 对拍（本机 4 核 Xeon，flex release；进程分开跑）

待 `compare_ort` / `breakdown` / `mem_stress` / `ort-mem` 填。
对外 512 用 compare_ort 的 `forward_raw`。不要用 Mac Python 4.3 / 201。
