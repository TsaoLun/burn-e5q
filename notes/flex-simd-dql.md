# AVX-512 DQL + flash `[B,1,1,S]` bias 融进 QK

> 2026-09-04。改动在 `vendor/burn-simd-dql`（`2b47a1b`），叠在 SIMD GELU 上。
> 对拍数字见 `notes/poc-results.md`「AVX-512 DQL」。
> 不换 A&S。不改 TILE。不挂钩 C-lite。不是再融单个 DQL codegen。

SIMD GELU 之后 512 `forward_raw` **269 ms / 5.0×** 本机 Rust ort 53.8 ms。
隔离里 DQL ×48 仍是 **33 ms**（36 个 `[512,384]` 低于 rayon 门槛，走串行标量）。

## 做了什么

1. **DQL AVX-512**  
   minmax 用 `_mm512_min/max`；quantize 是
   `round_ties_even(v / scale) + zp` 再 clamp 到 `[0,255]`
   （`_mm512_roundscale_ps` + `cvtepi32_epi8`）。公式和标量/`expanded`
   路径一致，单测 bit 对齐。大 buffer 仍 16K `par_chunks`。

2. **flash bias 融进 QK**  
   e5 的 attention bias 是 `[B,1,1,S]`（`q_step == 0`）。以前每个 query
   行标量加一遍；现在满 64 宽的 tile 在 QK 累加之后直接 `_mm512_add_ps`。
   TILE 仍是 64。C-lite 仍不挂钩。

3. **scratch 复用**  
   每个 rayon worker 一块 `[S×64]` scores，按 head 重置，不再每 head 分配。

## 单测

`cargo test -p burn-flex --release --lib -- dql attention`：34/34。

- 原有 DQL fused-vs-expanded + ties-to-even
- 新增 `[1,512,384]` / `[1,512,1536]` bit 对齐
- 隔离 `[512,1536]` DQL：AVX-512 **0.18 ms** vs 标量 **1.68 ms**
- 原有 D=32 flash vs naive / bias broadcast / partial tile

## 对拍（本机 4 核 Xeon，flex release；进程分开跑）

`compare_ort` 主表仍印 Mac Python 4.3 / 1412 / 201。分母用本轮单独
`ort-mem`（arena off）：短 **3.6** / packed **1201** / 512 **55.6**。
历史 3.5 / 1099 / 53.8 也列着，512 倍数差 0.1×。

| 口径 | SIMD GELU | **这一刀** | 本机 Rust ort | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok `forward_raw` | 10.9 | **10.1** | 3.6 | **2.8×** |
| packed batch `embed_passages` | 2401 | **1905** | 1201 | **1.6×**（burn 含 SP） |
| 512 `forward_raw` | 269 | **185** | 55.6 | **3.3×** |
| 512 vs 历史 53.8 | 269 | **185** | 53.8 | **3.4×** |
| 4×512 `mem_stress` | 3050 | **2827** | 514 | **5.5×** |

mean cos **0.9950**（min **0.9876**），ranking 2/2。
隔离：DQL ×48 **33 → 4.8 ms**；flash ×12 **132 → 78 ms**（bias 融进 QK + scratch）。
`forward_raw` **269 → 185**（−84 ms）。对外用 compare_ort，不用 breakdown 的 177。

`mem_stress -- 5 2048`：稳态 RSS **213 / 232 MB**。Rust ort 同预算 **195 / 348 MB**。

到 2×（512 ~108–111 ms）还差 ~75 ms。大头是 flash（78 / ~42%）和 MMI
（隔离 54，整网 AMX 更轻）。不要再调 TILE / 再挂钩 C-lite / 再融单个 DQL codegen。
