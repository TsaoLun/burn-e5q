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

`compare_ort` 主表仍印 Mac Python 4.3 / 1412 / 201。分母用本轮单独
`ort-mem`（arena off）：短 **2.4** / packed **1096** / 512 **52.4**。

| 口径 | AMX pack | **这一刀** | 本机 Rust ort | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok `forward_raw` | 3.2 | **2.5** | 2.4 | **1.0×** |
| packed batch `embed_passages` | 1501 | **1366** | 1096 | **1.2×**（burn 含 SP） |
| 512 `forward_raw` | 123.6 | **103.8** | 52.4 | **2.0×** |
| 4×512 `mem_stress` | 2609 | **2444** | 458 | **5.3×** |

mean cos **0.9952**（min **0.9903**）。ranking 1/2（第二条 top-1 仍中，2/3 互换）。
`forward_raw` **123.6 → 103.8**（−20 ms）。隔离 fused LN ×25 **21.8 → 1.6 ms**。

`mem_stress -- 5 2048`：稳态 RSS **234 / 257 MB**。Rust ort **196 / 350 MB**。
4×512 五轮 2337–2815，中位 2444。

512 已到本机 Rust ort 的 **~2×**（目标线 ~105 ms）。大头变成 flash（41）和
MMI / GELU。不要再调 TILE / 再挂钩 C-lite / 再融单个 DQL codegen。
下一刀：整层 FFN 融合，或再砍 flash。
