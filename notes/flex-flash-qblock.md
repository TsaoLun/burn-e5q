# Q-block D=32 flash（Br=16，不改 TILE）

> 2026-09-04。改动在 `vendor/burn-flash-qblock`（`4d580bc`），叠在 SIMD DQL 上。
> 对拍数字见 `notes/poc-results.md`「Q-block flash」。
> 不改 TILE。不挂钩 C-lite。不是再融单个 DQL codegen。

SIMD DQL 之后 512 `forward_raw` **185 ms / 3.3×** 本机 Rust ort 55.6 ms。
隔离 flash ×12 仍是 **78 ms（~42%）**。上一刀已经把 `[B,1,1,S]` bias 融进 QK。

## 做了什么

TILE（Bc / KV）**仍是 64**。改的是 FlashAttention 的 **Br（query block）**：

1. 每个 KV tile 只转置一次 K → `[32, 64]`
2. 按 **16 个 query** 一块做 QK → softmax → PV。scores 从 `512×64`（128 KiB）
   变成 `16×64`（4 KiB），留在 L1
3. 满 64 宽时 QK / PV 一次走 **4 行**，共用同一趟 K / V load
4. PV 去掉 `p == 0` 分支（softmax 几乎不会是精确 0，分支更亏）

C-lite 仍不挂钩。

## 单测

`cargo test -p burn-flex --release --lib -- attention`：23/23。

- 原有 d32 vs naive / gemm-flash / partial tile + bias
- 隔离 12h×512：d32 **1.86 ms/层** vs gemm-flash **6.6 ms/层**
- 带 `[1,1,1,S]` bias：1.63 ms/层
- B=8：14.2 ms（7.6×）

## 对拍（本机 4 核 Xeon，flex release；进程分开跑）

`compare_ort` 主表仍印 Mac Python 4.3 / 1412 / 201。分母用本轮单独
`ort-mem`（arena off）：短 **3.7** / packed **1168** / 512 **53.5**。

| 口径 | SIMD DQL | **这一刀** | 本机 Rust ort | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok `forward_raw` | 10.1 | **10.1** | 3.7 | **2.7×** |
| packed batch `embed_passages` | 1905 | **1686** | 1168 | **1.4×**（burn 含 SP） |
| 512 `forward_raw` | 185 | **149** | 53.5 | **2.8×** |
| 4×512 `mem_stress` | 2827 | **2938** | 498 | **5.9×** |

mean cos **0.9950**（min **0.9876**），ranking 2/2。
隔离 flash ×12：**78 → 40 ms**。`forward_raw` **185 → 149**（−36 ms）。
对外用 compare_ort 的 149，不用 breakdown 的 146。

`mem_stress -- 5 2048`：稳态 RSS **213 / 232 MB**。Rust ort 同预算 **195 / 348 MB**。
4×512 墙钟几乎没掉（B=4 一次 forward，噪声大于 Br 收益）。

到 2×（512 ~107 ms）还差 ~42 ms。大头变成 MMI（隔离 54）和 fused LN / GELU。
不要再调 TILE / 再挂钩 C-lite / 再融单个 DQL codegen。
