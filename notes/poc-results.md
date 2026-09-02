# 阶段 3 PoC 结果：burn int8 E5 vs ort

> 日期：2026-09-01。环境：Intel Mac（x86_64，非 ARM），flex 后端，release。
> 代码：`burn-e5q/crates/e5-embed`（lib + compare_ort / diagnose / mem_stress 三个 bin）。
> 参考数据：`scripts/gen_ref.py` 用 Python onnxruntime 跑同一个 int8 模型生成（`ref_data.json`）。

## TL;DR

| 维度 | 结果 | 判定 |
|---|---|---|
| 图导入 | int8 模型（96× DQL + 96× MatMulInteger，1630 节点）完整导入编译 | ✅ |
| tokenizer | 9/9 case 的 token ids 与 HF tokenizers 完全一致 | ✅ |
| 数值（int8） | cos 0.9935–0.9975（mean 0.9960），未达 0.999 目标 | ⚠️ 见分析 |
| 检索排序 | top-1 命中 2/2；top-3 严格排序 0/2（2/3 位交换） | ⚠️ 可接受边缘 |
| 延迟 | 慢 ort **19–45×** | ❌ 阶段 4 必需 |
| 内存 | 常驻 92.6MB ✓；4096 token 预算峰值 **640MB 超预算**；降 2048 预算 → 416MB | ⚠️ 需降预算或优化 |

## 数值分析：0.996 是 burn 的 bug 吗？

**不是。** 证据链：

1. FP32 模型（model-check）burn vs ort 在 1e-4 容差内**精确匹配**——图转换本身无误。
2. last_hidden 逐元素诊断（`diagnose` bin）：mean abs diff 0.021，恰好约等于一个 int8 量化步长（激活 range/255），差异遍布各 token/dim，不是稀疏异常点。
3. ONNX spec 对 DQL 的 round 明确是 ties-to-even，burn `Tensor::round()` 文档也是 ties-to-even ✓ 实现符合 spec。
4. [ORT issue #28609](https://github.com/microsoft/onnxruntime/issues/28609) 官方确认：int8 量化推理**跨实现无法 bit 一致**——ORT 自己在 ARM64（NEON `vcvtnq`）/ x64（SSE `_mm_cvtps`，依赖 MXCSR）/ WASM（`std::nearbyintf`）之间就有 ±1 量化单位的边界差异，且 MatMulInteger 在部分硬件有饱和问题；深层网络里逐层累积。我的参考基线（Python ort on x86_64 Mac）与线上（ort crate on x86_64 Linux VNNI）之间同样存在这类差异。

结论：0.996 cos 属 int8 跨引擎固有分歧量级。真正该盯的是**检索排序一致性**：top-1 全中，2/3 位在分数相近的文档间交换——上线前建议用真实语料跑一轮 recall@k 评估，而非看绝对 cos。

## 延迟明细（flex，x86_64 release）

| 场景 | burn | ort 基线 | 倍数 |
|---|---|---|---|
| 单条短文本（16 tok） | 192 ms | 4.3 ms | ~45× |
| 8 条 batch（含 512 长文，pack 后 2 个 forward） | 26.9 s | 1.41 s | ~19× |
| 单条 512 tok | 4.5 s | 201 ms | ~22× |

根因：`MatMulInteger` 在 flex 后端走 naive i32 三重循环（`burn-flex/src/ops/matmul.rs`），无 SIMD/无分块。ort 在 x86_64 上有 VNNI（`vpdpbusd` 一条指令做 4 元素 u8×i8 点积）。**这正是阶段 4（cubek i8 GEMM）要解决的问题。**

## 内存明细

| 阶段 | RSS |
|---|---|
| 进程启动 | 0.7 MB |
| 模型加载（int8 118MB 权重 → bpk mmap/加载） | 92.6 MB |
| 4096 预算 worst-case（8×512）首 forward | 609 MB |
| 10 轮后（棘轮，分配器不归还） | 640 MB |
| 2048 预算（4×512）峰值 | 416 MB ✓ |

- 常驻部分（93MB）很健康；超预算的是 **forward scratch**：attention 矩阵（8×12×512²×4B ≈ 100MB/层峰值）+ 96 个 MatMulInteger 的 i32 中间张量（FFN 中间层 8×512×1536×4B = 25MB/处）叠加，且 macOS 分配器不归还（与 inmotion-social 关闭 ORT arena 的问题同源）。
- 缓解：token 预算 4096→2048 即进预算（416MB），代价是 batch 拆分更细、吞吐略降。
- 根治方向：阶段 4 换 cubek kernel 后 scratch 形态会变；或给 flex/生成代码加中间张量及时释放（现在 forward 内所有中间值活到作用域结束）。

## 意外收获：上游 bug 修复

对拍暴露 `test_dynamicquantizelinear*_expanded`（DQL 展开成基础算子的官方测试）**编译失败**：`use of moved value`。根因是 `burn-onnx/src/import/burn/graph.rs` 的 `build_scope` 里，ScalarTensor 图输出在 `convert_graph_boundary_scalars` 后类型已变 `ScalarNative`，输出侧的 future-use 注册 filter 漏掉了它们（输入侧有 `boundary_input_conversions` 兜底，输出侧没有）→ clone 计数差 1 → use-after-move。

已修复（`63e35840`，[TsaoLun/burn-onnx](https://github.com/TsaoLun/burn-onnx) 分支 `add-dynamic-quantize-linear`），修复后 6 条 DQL 官方测试 + 全部 836 条官方测试编译通过。

## 已知环境限制

- 本机是 Intel Mac，`ort` crate（2.0-rc）无 x86_64-apple-darwin 预编译包 → `cargo test -p burn-onnx` 的 lib tests 本地跑不了，靠 CI 或 Linux。onnx-official-tests（不依赖 ort）可跑。
- 官方测试 harness 不支持 rank-0 scalar 输出 → DQL 的 6 条官方测试 codegen 编译通过但无数值 harness；数值正确性由 insta 快照 + dql-poc 端到端（`y=[153,255,0,25,187,178]` 与手算一致）+ 本对拍覆盖。

## 复现

```bash
# 参考数据（需代理）
uv run scripts/gen_ref.py
# 对拍 + 延迟 + 排序
cargo run --release -p e5-embed --bin compare_ort
# 逐元素诊断
cargo run --release -p e5-embed --bin diagnose
# 内存压测：./mem_stress <rounds> <token_budget>
cargo run --release -p e5-embed --bin mem_stress -- 10 4096
```

## 下一步（按优先级）

1. **flex 融合 DQL**（`notes/flex-dql.md`）：codegen 一行 + flex 两趟 kernel。对拍见本文件「融 DQL」。
2. **flex VNNI + 融合 zp**（`notes/flex-vnni.md`）：短 33 ms / 512 tok 1.46 s。
3. **flex 分块**（`notes/flex-accel.md`）：SSE2 下 130→76 ms / 3.82→2.80 s。被 native+VNNI 取代为默认路径。
4. **阶段 4 cubecl**：短文本仍被 launch 卡住；叶子还不是手写 VNNI。
5. 上线前用真实语料做 recall@k 评估（替代绝对 cos 阈值）。

---

# 阶段 4：cubecl-cpu i8 GEMM 对拍

> 日期：2026-09-02。环境：Linux x86_64 KVM，4 核 Xeon（`avx512_vnni` + `amx_int8`），rustc 1.98，`CXX=g++`。
> 命令：`cargo run --release -p e5-embed --features cpu --no-default-features --bin compare_ort`
> 栈：`notes/stage-4-impl.md`（vendor snapshot：cubek CpuGemm + burn `int_matmul`→I32 + burn-onnx 代数 zp + cubecl host TargetMachine）。

## TL;DR

| 维度 | 结果 | 判定 |
|---|---|---|
| 图导入 / 编译 | 1630 节点、96×DQL+96×MatMulInteger 在 cubecl-cpu 上编过、跑通 | ✅ |
| tokenizer | 9/9 与 HF ids 一致 | ✅ |
| 数值 | min cos 0.9905，mean 0.9953（flex 阶段 3 为 mean 0.9960） | ⚠️ 同属 int8 跨引擎分歧 |
| 检索 | top-1 2/2；top-3 严格顺序 0/2（与 flex 相同模式） | ⚠️ |
| 延迟 vs Mac `ref_data.json` ort | 短文本 **450×**（1936 vs 4.3 ms）；512 tok **16×**（3201 vs 201 ms） | ❌ 未达 ~2× 目标 |
| 延迟 vs 本仓库 flex（同机阶段 3 约 130 ms / 3.8 s） | 短文本更慢；长文本约 **1.2× 快于 flex** | ⚠️ 短序列被 launch 卡住 |

叶子仍是 tiled `SUM_PROD` + LLVM host TM 自动向量化，**不是**手写 `vpdpbusd`。本机 ort/AMX 基线（约 3.4 ms / 50 ms）更远。

## 延迟明细（cubecl-cpu，release，进程内 3 次取 min）

对拍里 cosine 段已用过同一条短中文，所以 1936 ms 是 **稳态**，不是首次 JIT。

| 场景 | cubecl-cpu | Mac ort（`ref_data.json`） | 倍数 |
|---|---|---|---|
| 单条短文本（16 tok） | 1936 ms | 4.3 ms | ~450× |
| 8 条 batch（含 512 长文） | 7232 ms | 1412 ms | ~5.1× |
| 单条 512 tok | 3201 ms | 201 ms | ~16× |

模型加载 253 ms，RSS 加载后 241 MB，全部推理后 **665 MB**。

## 内存（`mem_stress -- 5 2048`）

| 阶段 | RSS |
|---|---|
| 启动 | 40 MB |
| 模型加载 | 241 MB |
| 4×512 首 round（含 JIT） | 515 MB / 8037 ms |
| round 1–4 稳态 | 516 MB / ~6.2 s |

2048 token 预算峰值 **515.5 MB**，刚过 512 MB 容器线（flex 同预算 416 MB）。CubeCL scratch + zp 的 i32 `sum_dim` 更肥。4096 预算未再测（compare_ort 全流程已到 665 MB）。

## 为什么没打平 ort

1. **Launch 开销**：图约 1630 个节点。每个 MatMulInteger 现在是 `u8/i8 GEMM` **再加** `sum_dim`×2 + cast/mul/sub/add（代数 zp）。短序列（K=32、M≈16）上这些 CubeCL-CPU launch 比 flex 的进程内 i32 三重循环还贵。
2. **GEMM 叶子不是 AMX/VNNI intrinsic**：host `TargetMachine` 只是让 LLVM O3 有机会 autovec；ort 走的是 AMX/`vpdpbusd`。
3. **96 个 DQL** 仍在 float 路径，没有和 GEMM 融合。

数值没有崩：mean cos 0.995 说明 u8×i8→i32 + 代数 zp 与 ort 同量级，问题在调度/内核质量，不在语义。

## 下一步（性能）

1. 把 zp 补偿融进同一个 integer GEMM（`(A-za)@(B-zb)` 一次 launch），砍掉每层两次 `sum_dim`。
2. 给 8-bit 叶子真正的 VNNI/AMX 微内核（或确认 LLVM 已打出 `vpdpbusd`）。
3. 有写权限后把 `vendor/*` 迁回 TsaoLun/{cubek,burn,burn-onnx,cubecl} 真分支。

---

# flex 加速：分块 u8/i8/i32 GEMM

> 日期：2026-09-02。同一台 4 核 Xeon。`vendor/burn-route-int8-matmul` `005354fd`。
> 命令：`cargo run --release -p e5-embed --bin compare_ort`（默认 flex）。
> 实现：`notes/flex-accel.md`。codegen 仍是代数 zp（与 cubecl 共用）。

## TL;DR

| 维度 | 结果 | 判定 |
|---|---|---|
| 数值 | min cos 0.9935，mean **0.9960**（与阶段 3 flex 一致） | ✅ |
| 检索 | top-1 2/2；top-3 严格顺序 0/2 | ⚠️ 同前 |
| 短文本 | **75.7 ms**（本机旧 flex 130 ms → **1.7×**） | ✅ |
| 512 tok | **2797 ms**（旧 3.82 s → **1.4×**） | ✅ |
| vs Mac ort | 短 18×，512 **14×**（旧 30× / 19×） | 仍未达 ~2× |
| vs 本机 AMX ort | 短 ~22×，512 ~56× | 需要 VNNI/融合 |
| vs cubecl-cpu | 短 1936→76 ms；512 3201→2797 ms | flex 仍是短查询正解 |

模型加载 150 ms，RSS 88 MB；全流程后 **254 MB**（cubecl 同流程 665 MB）。
`mem_stress -- 5 2048`：峰值 **230 MB**（进 512 MB 预算；cubecl 同预算 516 MB）。4×512 稳态约 11.7 s（cubecl ~6.2 s，大 batch 仍是 cubecl 更快）。

## 延迟明细

| 场景 | 本机旧 flex | 本机新 flex | cubecl-cpu | Mac ort | 本机 ort |
|---|---:|---:|---:|---:|---:|
| 16 tok | 130 ms | **75.7 ms** | 1936 ms | 4.3 ms | 3.4 ms |
| 8 条 batch | 26.2 s | **18.1 s** | 7.2 s | 1.41 s | 0.53 s |
| 512 tok | 3.82 s | **2.80 s** | 3.20 s | 201 ms | 50 ms |

8 条 batch 上 cubecl 仍更快（更大 `M` 摊薄 launch）。单条短/长查询 flex 更好。

## 还剩什么

内核还不是 `vpdpbusd`；96 个 DQL 和代数 zp 的 `sum_dim` 仍在。短文本再往下会碰到这些固定开销，不是再改一版三重循环能解决的。

下一步见 `notes/flex-vnni.md`。

---

# flex VNNI + 融合 zp

> 日期：2026-09-02。环境：同机 4 核 Xeon（`avx512_vnni`）。
> 栈：`vendor/burn-route-int8-matmul` `30cab97` + `vendor/burn-onnx-keep-int8-matmul` `3a2bf47` + `.cargo/config.toml` `target-cpu=native`。
> 命令：`cargo run --release -p e5-embed --bin compare_ort`
> 实现：`notes/flex-vnni.md`。

## TL;DR

| 维度 | 结果 | 判定 |
|---|---|---|
| 数值 | min cos 0.9935，mean **0.9960** | ✅ 与分块 flex 一致 |
| 检索 | top-1 2/2；top-3 严格顺序 0/2 | ⚠️ 同前 |
| 短文本 | **33.2 ms**（分块 75.7 → **2.3×**；朴素 130 → **3.9×**） | ✅ |
| 512 tok | **1458 ms**（分块 2.80 s → **1.9×**） | ✅ 贴近非 GEMM 下限 |
| vs Mac ort | 短 **7.7×**，512 **7.3×**（分块时 18× / 14×） | 仍未达 ~2× |
| vs 本机 AMX ort | 短 ~10×，512 ~29× | DQL / 非 GEMM 为主 |
| 内存 | 加载 88 MB；`mem_stress 2048` **215 MB** | ✅ |

生成代码已是 `.matmul_integer(...)`，不再对每个 MatMulInteger 展开 `sum_dim`。

## 延迟明细

| 场景 | 朴素 flex | 分块 flex | **VNNI+zp** | cubecl-cpu | Mac ort | 本机 ort |
|---|---:|---:|---:|---:|---:|---:|
| 16 tok | 130 ms | 75.7 ms | **33.2 ms** | 1936 ms | 4.3 ms | 3.4 ms |
| 8 条 batch | 26.2 s | 18.1 s | **8.73 s** | 7.2 s | 1.41 s | 0.53 s |
| 512 tok | 3.82 s | 2.80 s | **1.46 s** | 3.20 s | 201 ms | 50 ms |

`mem_stress -- 5 2048`：4×512 稳态 **~6.4 s**（分块 11.7 s），峰值 215 MB。

## 还剩什么

512 tok 的 1.46 s 已经靠近「96 个 DQL + LN + clone」的固定开销。下一步见本文件「融 DQL」。

---

# 融 DQL

> 日期：2026-09-02。栈：`vendor/burn-route-int8-matmul` `2d1084f` + `vendor/burn-onnx-keep-int8-matmul` `7cc2d36`。
> 实现：`notes/flex-dql.md`。对拍数字在 `compare_ort` / `mem_stress` 跑完后填入。

待测：性能（短 / batch8 / 512）、吞吐（q/s、tok/s）、内存（加载 RSS + `mem_stress -- 5 2048`）。
