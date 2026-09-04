# AVX-512 GELU / flash ILP（区间提前返回 + 4 行 softmax）

> 2026-09-04。改动在 `vendor/burn-flash-gelu-ilp`（`a10b2c0`），叠在 FFN 融合上。
> 对拍数字见 `notes/poc-results.md`「GELU/flash ILP」。
> 不改 TILE。不挂钩 C-lite。不融单个 DQL。不用 A&S 换默认 erf。

FFN 融合之后 512 `forward_raw` **106.1 ms / 2.7×** 本机 Rust ort 39.3 ms。
breakdown：flash ×12 **41 ms（~39%）**，隔离 fused GELU ×12 **19 ms**。
多项式仍是 musl/fdlibm，不是 Abramowitz–Stegun。

## 做了什么

1. **GELU erf 均匀区间提前返回 + 双 zmm**  
   整条 zmm 都 `|x|≥6` → `copysign(1)`；都 `|x|<0.84375` → 只算 small poly。
   混区间仍按 lane 算需要的那一段。`gelu_ptr` 一次走两个独立 zmm。
   融合 `dequant_affine_gelu` 继续一次一条 zmm（双 zmm 在融合路径上更慢）。

2. **flash 4 行 softmax 交错**  
   e5 路径（`mask=None`，`[B,1,1,S]` bias 已融进 QK，满 TILE）一次处理
   4 行：max / Cephes exp / output rescale 交错，盖住 `exp` 延迟。
   某行 `tile_max == -inf` 时回退到逐行，避免 `exp(x - (-inf))`。
   Br 仍是 16，TILE 仍是 64。

3. **8×8 AVX K 转置**  
   满 TILE 的 `k_t[d*TILE + ki] = k[(kv_start+ki)*D + d]` 用 ymm
   `unpack` + `shuffle` + `permute2f128`。短尾巴仍走标量。

`unsafe` 只在 burn-flex。burn-e5q / burn-onnx 无 `unsafe`。

## 单测

`cargo test -p burn-flex --release --lib -- gelu dequant_affine attention`：37/37。

- 新增：K-tile 转置 vs 标量；4 行 softmax vs 逐行（含一行全 `-inf`）
- 隔离 `[512,1536]` GELU：AVX-512 **0.72 ms** vs 标量 **3.96 ms**；×12 ≈ 8.6 vs 47.5
- 隔离 e5-like 12h×512 flash：**0.9 ms / 层**（×12 ≈ 11；带 bias 0.8）
  上一刀同口径大约 1.8–3.4 ms / 层、端到端 flash 41 ms。隔离和整网缓存不是一回事。

## 对拍（本机 4 核 Xeon，flex release；进程分开跑）

`compare_ort` 主表仍印 Mac Python 4.3 / 1412 / 201。分母用本轮单独
`ort-mem`（arena off）：短 **2.4** / packed **1040** / 512 **39.2**。

| 口径 | FFN fuse | **这一刀** | 本机 Rust ort | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok `forward_raw` | 2.5 | **2.7** | 2.4 | **1.1×** |
| packed batch `embed_passages` | 1398 | **1504** | 1040 | **1.4×**（burn 含 SP） |
| 512 `forward_raw` | 106.1 | **102.0** | 39.2 | **2.6×** |
| 4×512 `mem_stress` | 2286 | **2343** | 355 | **6.6×**（burn 含 SP） |

mean cos **0.9952**（min **0.9903**）。ranking 1/2（第二条 top-1 仍中，2/3 互换）。
`compare_ort` 512 **102.0**；breakdown 校准 **95.7**。相对 FFN 刀大约 −4 ms。
隔离 flash ×12 **41 → 34.5**；隔离 fused GELU ×12 仍是 **18.5**（生成图 FFN1 走融合 dequant，不走独立 GELU）。

`mem_stress -- 5 2048`：五轮 2337–2514，中位 **2343**。RSS **234 / 257 MB**。
Rust ort 同预算 **195 / 349 MB**，4×512 中位 **355**。

端到端没掉隔离里那种 0.9 ms/层。flash 在整网里仍是缓存墙；MMI 72 仍是 **47.6 ms**。
不要再调 TILE / 再挂钩 C-lite / 再融单个 DQL codegen。下一刀再砍 MMI。
