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

> 日期：2026-09-02。同一台 4 核 Xeon（`avx512_vnni`）。
> 栈：`vendor/burn-route-int8-matmul` `2d1084f` + `vendor/burn-onnx-keep-int8-matmul` `7cc2d36`。
> 命令：`cargo run --release -p e5-embed --bin compare_ort`；`mem_stress -- 5 2048`；`cargo run --release -p ort-mem -- -- 5 2048`。
> 基线：本机 Rust `ort` 2.0.0-rc.13，CPU EP，arena off（inmotion-social）。不用 Python onnxruntime。
> 实现：`notes/flex-dql.md`。

## TL;DR

只把 DQL 收成一个 kernel（minmax + 一趟量化），没有把 DQL+MMI+反量化打成一个 op。短句几乎不动；512 从 1.46 s 掉到 **1.27 s**（约 13%），落在先前估的档位 A。

| 维度 | 结果 | 判定 |
|---|---|---|
| 数值 | min cos 0.9935，mean **0.9960** | ✅ 与 VNNI+zp 一致 |
| 检索 | top-1 2/2；top-3 严格顺序 0/2 | ⚠️ 同前 |
| 短文本 | **29.6 ms**（VNNI 33.2 → **1.12×**） | 调度主导，符合预期 |
| 8 条 batch | **7.07 s**（8.73 → **1.23×**） | ✅ |
| 512 tok | **1272 ms**（1458 → **1.15×**） | ✅ |
| vs 本机 Rust ort | 短 **12×**（2.4 ms），packed batch **7.6×**（936 ms），512 **24×**（53.8 ms） | 未达 ~2× |
| 内存 | 4×512 稳态 **213 / 315** vs Rust ort **162 / 268** | ✅ 进 512 MB；大约 +51 MB RSS |

`dql-poc` 官方输入：`y_scale=0.019607844`，`zp=153`，`y` 以 `[153, 255, 0, …]` 开头。

## 延迟

`ort-mem` 与 `compare_ort` 用同一份 `ref_data.json` token ids。packed batch 走和 burn 一样的 4096 token `pack_batches`（7 条非空 passage：一条 512 + 短句）。padded 是一次 pad 到最长的 7×512，ORT 自己也能这么跑。

| 场景 | 朴素 flex | 分块 | VNNI+zp | **融 DQL** | **Rust ort arena off** | 倍数 |
|---|---:|---:|---:|---:|---:|---:|
| 16 tok | 130 ms | 75.7 ms | 33.2 ms | **29.6 ms** | **2.4 ms** | **12×** |
| 7 条 packed | 26.2 s | 18.1 s | 8.73 s | **7.07 s** | **936 ms** | **7.6×** |
| 7 条 padded 7×512 | — | — | — | — | 946 ms | — |
| 512 tok | 3.82 s | 2.80 s | 1.46 s | **1.27 s** | **53.8 ms** | **24×** |

arena on 时 ORT 短句 2.3 ms、packed 516 ms、512 **52.6 ms**（更快，但内存见下）。Mac 上 Python ort 写进 `ref_data.json` 的 4.3 / 1412 / 201 ms 不再当基线。

Rust ort（rc.13 自带的 ORT）对 `ref_data.json` 里 Python 1.29 向量的 mean cos 也是 **0.9968**（min 0.9956）。burn vs 那份 ref 的 0.996 和「两个 ORT 构建之间」是同一量级，不是 burn 独有的漂移。

## 吞吐（本机 flex vs 本机 Rust ort，arena off）

| 场景 | 融 DQL q/s | 融 DQL tok/s | Rust ort q/s | Rust ort tok/s |
|---|---:|---:|---:|---:|
| 16 tok | **33.8** | **540** | **414** | **6625** |
| packed 7 条（593 tok） | **1.13** | **85** | **7.48** | **633** |
| 512 tok | **0.79** | **403** | **18.6** | **9510** |

## 内存：burn vs 整进程 Rust ort

口径：同一台 4 核 Xeon、同一份 ONNX（磁盘 112.9 MB）。数字是 **进程 VmRSS / VmHWM**。ORT 列来自 `cargo run --release -p ort-mem`（Rust `ort`，不链 Burn）。生产是 **arena off**。

| 阶段 | burn 融 DQL | **Rust ort arena off** | Rust ort arena on |
|---|---:|---:|---:|
| 进程启动 | 3.1 | 7.1 | 7.1 |
| 运行时库就绪 | （链进二进制） | 16.3 | 16.5 |
| 模型加载后 RSS | 87.6 | 153.1 | 154.0 |
| 加载冲高 HWM | 97.4 | 234.9 | 235.3 |
| 4×512 单独压测 RSS | 213.4 | **162.2** | 404.2 |
| 4×512 单独压测 HWM | 315.1 | **267.8** | 404.2 |
| compare 全流程 RSS | 236.1 | **193.1** | 564.1 |
| compare 全流程 HWM | 401.0 | **346.4** | 564.1 |

4×512 单独一行来自只跑 `mem_stress` 形状的进程（burn：`mem_stress -- 5 2048`；ORT：上一轮纯压测）。compare 全流程是同一进程里跑完 cosine + 短/packed/padded/512 之后的占用（ORT 的 padded 7×512 会把 HWM 抬到 346）。

读法：

1. **容器只看整进程。** arena off：4×512 稳态 burn **213 vs 162**（+51 MB），HWM **315 vs 268**（+47 MB）。compare 全流程 HWM **401 vs 346**。两边都进 512 MB。
2. **加载更轻的是 burn**（88 vs 153）。ORT 加载会冲到 HWM 235 再回落到 ~153。
3. **scratch 更重的是 burn。** 4×512 HWM 相对加载后：burn +227 MB，ORT +114 MB。
4. **arena on 不能当生产对照。** 单独 4×512 已到 404 MB；把 compare 的 7×512 叠进去会到 **564–590 MB**，超过 512 MB 容器线。这就是线上关 arena 的原因。

4×512 时延：burn ~5.8 s；Rust ort arena off ~0.4–0.7 s（看进程里是否已经分配过 scratch）。

## 还剩什么

512 tok 仍有 ~1.2 s 不是 GEMM 也不是 DQL：softmax、LayerNorm、eager 调度。本机 Rust ort 512 是 **54 ms**，要到 ~2× 得进 ~100 ms。下一步见本文件「融合 attention」。

---

# 融合 attention

> 日期：2026-09-03。同一台 4 核 Xeon（`avx512_vnni`）。
> 栈：`vendor/burn-onnx-coalesce-int8-attn` `f78e156` + `vendor/burn-flash-512` `245ab35`。
> 命令：`cargo run --release -p e5-embed --bin compare_ort`；`mem_stress -- 5 2048`。
> 基线：上一节本机 Rust ort arena off（2.4 / 936 / 53.8 ms）。实现：`notes/flex-attn.md`。

## TL;DR

图改对了：12 层都收成 `attention()`，Softmax 清零，DQL 96→48，MMI 96→72，512 走 flex flash。
**延迟判断被证伪。** 512 只从 1.27 s 掉到 **1.15 s**（10%），没进 300–500 ms 分水岭。
`[H,S,S]` 不再物化（4×512 HWM 315→278），所以剩下的 ~21× 不是 score 矩阵。

| 维度 | 结果 | 判定 |
|---|---|---|
| 图 | 12× `module::attention`，0 Softmax，scale=`1/√32` | ✅ |
| 数值 | min cos 0.9861，mean **0.9946** | ⚠️ 比融 DQL 的 0.9960 略漂 |
| 检索 | top-3 **2/2**（融 DQL 是 0/2） | ✅ 排序反而齐了 |
| 短文本 | **29.5 ms**（29.6 → 1.00×） | 调度主导，符合预期 |
| packed batch | **5.57 s**（7.07 → 1.27×） | 小 |
| 512 tok | **1154 ms**（1272 → 1.10×） | ❌ 未达分水岭 |
| vs 本机 Rust ort | 短 **12×**，batch **5.9×**，512 **21×**（53.8 ms） | 未达 ~2× |
| 内存 | 4×512 稳态 **213 / 278**（融 DQL 213 / 315） | ✅ HWM −37 MB |

## 延迟

| 场景 | 朴素 flex | VNNI+zp | 融 DQL | **融合 attn** | **Rust ort** | 倍数 |
|---|---:|---:|---:|---:|---:|---:|
| 16 tok | 130 ms | 33.2 ms | 29.6 ms | **29.5 ms** | **2.4 ms** | **12×** |
| packed batch | 26.2 s | 8.73 s | 7.07 s | **5.57 s** | **936 ms** | **5.9×** |
| 512 tok | 3.82 s | 1.46 s | 1.27 s | **1.15 s** | **53.8 ms** | **21×** |

吞吐：短 **33.9 q/s / 542 tok/s**；512 **0.87 q/s / 444 tok/s**。

## 内存

| 阶段 | 融 DQL | **融合 attn** | Rust ort arena off |
|---|---:|---:|---:|
| 加载后 RSS / HWM | 87.6 / 97.4 | **87.5 / 97.3** | 153 / 235 |
| 4×512 稳态 / HWM | 213 / 315 | **213 / 278** | 162 / 268 |
| compare 全流程 RSS / HWM | 236 / 401 | **238 / 336** | 193 / 346 |

4×512 时延：burn ~4.9 s（融 DQL ~5.8 s）。

## 还剩什么

`[H,S,S]` 不是剩下那 20×。下一刀是整层执行单元（QKV+attn+FFN），或让 flash 不慢于已经在跑的 VNNI QK。再融单个 DQL / Softmax 只会再啃约 10%。

---

# flash 按 head 并行

> 日期：2026-09-03。同一台 4 核 Xeon。
> 栈：`vendor/burn-flash-par-heads` `fd4f793`（叠在融合 attn 上）。
> 实现：`notes/flex-flash-par.md`。

## TL;DR

flash 的 12 个 head 改成 rayon 并行。数值与串行 flash 完全一致。
**512 1.15 s → 1.12 s（3%）**，4×512 HWM 仍是 278 MB。
`embed_passages` 的 1.12 s **含 sentencepiece**；模型本体见下一节。不要再调 TILE / gemm 并行。

| 场景 | 融合 attn | **head 并行** | Rust ort | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok | 29.5 ms | **28.4 ms** | 2.4 ms | **12×** |
| packed batch | 5.57 s | **5.44 s** | 936 ms | **5.8×** |
| 512 tok（含 tokenize） | 1.15 s | **1.12 s** | 53.8 ms | **21×**（不公平） |

---

# 拆 1.12 s

> 日期：2026-09-03。同一台 4 核 Xeon。
> 命令：`cargo run --release -p e5-embed --bin breakdown`
> 全文：`notes/gap-breakdown.md`。

`embed_passages` 1099 ms = sentencepiece **457 ms** + `forward_raw` **639 ms**。
Rust ort 53.8 ms 只有 `session.run`。公平模型倍数是 **12×**，不是 21×。
639 ms 被隔离块加总对上（差 1%）：MMI 228（36%）+ flash 205（32%）+ 展开 GELU 117（18%）+ LN 44 + DQL 33。

下一刀打模型内这三块（整层融合 / int8 flash / GEMM），不要再当「生成图税」或再调 flash TILE。`compare_ort` 的 512 行已拆 tokenize / `forward_raw`。

---

# 融 GELU / LayerNorm

> 日期：2026-09-03。同一台 4 核 Xeon。
> 栈：`vendor/burn-onnx-coalesce-gelu-ln` `68153cc` + `vendor/burn-flex-par-gelu` `319336c`。
> 实现：`notes/flex-gelu-ln.md`。

codegen 把展开 erf-GELU / 最后一维 LN 收成 `activation::gelu` 和 `nn::LayerNorm`。
flex 大 GELU 走 rayon（仍是 `libm::erff`）。

**做成了，整网没掉。** 生成图 12× `gelu` / 25× `LayerNorm` / 0 `erf`。mean cos **0.9950**。
隔离融合 GELU 83 vs 117、LN 22 vs 44（可省 ~56 ms），但 `forward_raw` **636 vs 639**（噪声）。
预期 480–520 ms 落空。读数用 `forward_raw`，不要用含 sentencepiece 的 `embed_passages`。

| 场景 | 融 attn + head 并行 | **融 GELU/LN** | Rust ort | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok | 28.4 ms | **28.6 ms** | 2.4 ms | **12×** |
| packed batch | 5.44 s | **5.60 s** | 936 ms | **6.0×** |
| 512 `forward_raw` | 639 ms | **636 ms** | 53.8 ms | **11.8×** |

下一刀：路线 C 整数 flash（模型 32%）和 MMI 再快（36%）。不要再调 TILE / 再融单个 DQL。

---

# 整数 flash + AMX

> 日期：2026-09-03。同一台 4 核 Xeon（`avx512_vnni` + `amx_int8`）。
> 栈：`vendor/burn-int8-flash-amx` `21dba0c`（叠在融 GELU/LN 上）。
> 实现：`notes/flex-int8-flash-amx.md`。

**AMX 做成了；C-lite 整数 QK 写成了但挂钩更慢，已拔掉。**

对齐的 72× MMI：228 ms @ 44 GOPS → **21 ms @ 280–600 GOPS（~11×）**。
VNNI QK + 物化 `[S,S]`：flash 205 → **280 ms**，所以 `attention_flash` 仍走 tiled f32。
`forward_raw` **636 → 414 ms**。mean cos **0.9950**。
对本机 Rust ort 的活数见下一节（短 3.8× / 512 **7.7×**）。不要用 `compare_ort` 主表里的 Mac Python 4.3 / 201。

`mem_stress -- 5 2048`：4×512 稳态 **212 / 275 MB**，约 3.95 s/round（进 512 MB）。

模型里现在最大的是 f32 flash（208 ms / 50%）和 fused GELU 的 `erff`（~83 ms）。
到 2× 还差 ~300 ms。不要再挂钩这版 C-lite，不要再调 TILE。

---

# 本机再对 Rust ort（AMX 之后）

> 日期：2026-09-04。同一台 4 核 Xeon。**两个进程分开跑**（叠着跑会把 ORT 512 从 54 抬到 84）。
> burn：`compare_ort` / `mem_stress -- 5 2048`（`vendor/burn-int8-flash-amx` `21dba0c`）。
> Rust ort：`ort-mem` arena off，4 intra-op，预编码 `ref_data.json` ids。

`compare_ort` 主表里的 4.3 / 1412 / 201 仍是 Mac Python ort，**不要当倍数分母**。下面用刚跑的 `ort-mem`。

## 延迟（公平 = 预编码 ids）

| 场景 | burn | **Rust ort arena off** | 倍数 | 口径 |
|---|---:|---:|---:|---|
| 16 tok `forward_raw` | **13.2 ms** | **3.5 ms** | **3.8×** | 只有模型 |
| 16 tok `embed_passages` | 16.8 ms | 3.5 ms | 4.8× | burn 含 3.3 ms SP |
| packed 7 条 | 3829 ms | **1099 ms** | 3.5× | burn 含 SP；ort 只有 session |
| 512 `forward_raw` | **414 ms** | **53.8 ms** | **7.7×** | 只有模型 |
| 512 `embed_passages` | 873 ms | 53.8 ms | 16.2× | burn 含 ~457 ms SP |
| 4×512 `mem_stress` | 3620 ms | **458 ms** | **7.9×** | dummy ids，无 SP |

512 的 53.8 ms 和融 DQL 时记下的数一致。短句这次是 3.5 ms（当时 2.4）；packed 1099（当时 936）。用这次的活数。

## 数值

| | burn vs Python ref | Rust ort vs 同一份 ref |
|---|---:|---:|
| mean cos | **0.9950** | **0.9968** |
| min cos | 0.9886 | 0.9956 |
| ranking top-3 | 2/2 | — |

跨引擎仍是 int8 固有分歧，不是 AMX 算错。

## 内存（整进程 VmRSS / VmHWM）

| 阶段 | burn AMX | **Rust ort arena off** |
|---|---:|---:|
| 启动 | 3.1 / 3.2 | 7.2 / 7.2 |
| 模型加载后 | **87.4 / 97.2** | 156 / 235 |
| 4×512 稳态 / HWM | **212 / 275** | **196 / 350** |
| compare 全流程 | 236 / 335 | 196 / 350 |

两边都进 512 MB。加载更轻的是 burn；4×512 HWM 也是 burn 更低（275 vs 350）。稳态 RSS burn 大约 +16 MB。

---

# D=32 AVX-512 flash

> 日期：2026-09-04。同一台 4 核 Xeon。
> 栈：`vendor/burn-flash-d32` `219fe61`（叠在 AMX 上）。
> 实现：`notes/flex-flash-d32.md`。
> 两个进程分开跑。分母是本机 Rust ort 3.5 / 1099 / 53.8，不是 Mac Python 4.3 / 201。

长序列 D=32 走 AVX-512 QK / softmax / PV；`[1,1,1,S]` 不再展开成 `[H,S,S]`。
C-lite 仍不挂钩。packed `[7,512]` 必须开 FTZ+DAZ，否则 denormal ~19 s。

mean cos **0.9950**（min 0.9876），ranking 2/2。

| 场景 | AMX + gemm flash | **D=32** | Rust ort | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok `forward_raw` | 13.2 | **13.9** | **3.5** | **4.0×** |
| packed 7 条 `embed_passages` | 3829 | **2928** | **1099** | **2.7×** |
| 512 `forward_raw` | 414 | **350** | **53.8** | **6.5×** |
| 512 `embed_passages` | 873 | **804** | 53.8 | 含 ~451 ms SP |

隔离 flash ×12：**208 → 129 ms**。`forward_raw` **414 → 350**（−64 ms）。
`mem_stress -- 5 2048`：4×512 **3511 ms**，RSS **213 / 232 MB**。

到 2×（512 ~108 ms）还差 ~240 ms。大头变成 fused GELU `erff`（~82）和整层融合。
不要再挂钩 C-lite，不要再调 TILE。

---

# AVX-512 GELU

> 日期：2026-09-04。同一台 4 核 Xeon。
> 栈：`vendor/burn-simd-gelu` `a62f534`（叠在 D=32 flash 上）。
> 实现：`notes/flex-gelu-simd.md`。
> 两个进程分开跑。分母是本机 Rust ort 3.5 / 1099 / 53.8。

连续 f32 GELU 走 musl/fdlibm `erff` 的 AVX-512 分段有理式。不是 A&S。
unary `erf` 仍是 `libm::erff`。上一刀 codegen 融 GELU 整网没掉；这次
in-place 和 alloc 都走 SIMD，整网吃到了。

mean cos **0.9950**（min 0.9876），ranking 2/2。

| 场景 | D=32 flash | **SIMD GELU** | Rust ort | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok `forward_raw` | 13.9 | **10.9** | **3.5** | **3.1×** |
| packed batch `embed_passages` | 2928 | **2401** | **1099** | **2.2×** |
| 512 `forward_raw` | 350 | **269** | **53.8** | **5.0×** |
| 512 `embed_passages` | 804 | **731** | 53.8 | 含 ~451 ms SP |

隔离 fused GELU ×12：**83 → 17 ms**。`forward_raw` **350 → 269**（−81 ms）。
`mem_stress -- 5 2048`：4×512 **3050 ms**，RSS **213 / 232 MB**。

到 2× 还差 ~161 ms。下一刀 AVX-512 DQL 已做（见下节）。

---

# AVX-512 DQL + flash bias 融进 QK

> 日期：2026-09-04。同一台 4 核 Xeon。
> 栈：`vendor/burn-simd-dql` `2b47a1b`（叠在 SIMD GELU 上）。
> 实现：`notes/flex-simd-dql.md`。
> 两个进程分开跑。不要用 Mac Python 4.3 / 201。

DQL minmax / quantize 走 AVX-512，公式仍是 `v / scale` + ties-to-even
（不能改成 `v * (1/scale)`，不能开 FTZ）。`[B,1,1,S]` bias 在 QK 满 64
宽 tile 的 epilogue 里加；worker 复用 `[S×64]` scratch。TILE 仍是 64。

mean cos **0.9950**（min 0.9876），ranking 2/2。

| 场景 | SIMD GELU | **DQL + QK bias** | Rust ort（本轮） | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok `forward_raw` | 10.9 | **10.1** | **3.6** | **2.8×** |
| packed batch `embed_passages` | 2401 | **1905** | **1201** | **1.6×** |
| 512 `forward_raw` | 269 | **185** | **55.6** | **3.3×** |
| 512 vs 历史 53.8 | 269 | **185** | 53.8 | **3.4×** |
| 512 `embed_passages` | 731 | **718** | 55.6 | 含 ~532 ms SP |

隔离 DQL ×48：**33 → 4.8 ms**。隔离 flash ×12：**132 → 78 ms**。
`forward_raw` **269 → 185**（−84 ms）。compare_ort 印 184.9；breakdown 校准 177。
`mem_stress -- 5 2048`：4×512 **~2827 ms**，RSS **213 / 232 MB**。

---

# 本机再对 Rust ort（SIMD DQL 之后）

> 日期：2026-09-04。同一台 4 核 Xeon。**两个进程分开跑**。
> burn：`compare_ort` / `mem_stress -- 5 2048` / `breakdown`（`vendor/burn-simd-dql` `2b47a1b`）。
> Rust ort：`ort-mem -- -- 5 2048`，arena off，4 intra-op，预编码 `ref_data.json` ids。

`compare_ort` 主表里的 4.3 / 1412 / 201 仍是 Mac Python ort，**不要当倍数分母**。
下面用刚跑的 `ort-mem`。packed：burn 是 8 条 passage（含空串，598 tok，含 SP）；
ort 是 7 条非空（593 tok，只有 session）。512 行主导两边。

## 延迟（公平 = 预编码 ids / 只有模型）

| 场景 | burn | **Rust ort arena off** | 倍数 | 口径 |
|---|---:|---:|---:|---|
| 16 tok `forward_raw` | **10.1 ms** | **3.6 ms** | **2.8×** | 只有模型 |
| 16 tok `embed_passages` | 14.2 ms | 3.6 ms | 3.9× | burn 含 3.9 ms SP |
| packed batch | 1905 ms | **1201 ms** | 1.6× | burn 含 SP；ort 只有 session |
| 512 `forward_raw` | **185 ms** | **55.6 ms** | **3.3×** | 只有模型 |
| 512 `embed_passages` | 718 ms | 55.6 ms | 12.9× | burn 含 ~532 ms SP |
| 4×512 `mem_stress` | 2827 ms | **514 ms** | **5.5×** | dummy ids，无 SP |

本轮 ort 512 是 55.6（历史多次 53.8）。对 2× 目标仍按 ~54 ms → **~108 ms**，还差 ~77 ms。

## 吞吐（同一组活数）

| 场景 | burn q/s | burn tok/s | Rust ort q/s | Rust ort tok/s |
|---|---:|---:|---:|---:|
| 16 tok 模型 | **99** | **1584** | **278** | **4455** |
| packed batch | 4.2 | 314 | 5.8 | 494 |
| 512 模型 | **5.4** | **2770** | **18.0** | **9201** |
| 4×512 | 1.4 | 725 | 7.8 | 3984 |

packed / `embed_passages` 吞吐被 sentencepiece 拉低，不拿来当模型基线。

## 数值

| | burn vs Python ref | Rust ort vs 同一份 ref |
|---|---:|---:|
| mean cos | **0.9950** | **0.9968** |
| min cos | 0.9876 | 0.9956 |
| ranking top-3 | 2/2 | — |

跨引擎仍是 int8 固有分歧，不是 DQL SIMD 算错。

## 内存（整进程 VmRSS / VmHWM）

| 阶段 | burn DQL | **Rust ort arena off** |
|---|---:|---:|
| 启动 | 3.1 / 3.2 | 7.2 / 7.2 |
| 模型加载后 | **87.4 / 97.3** | 155 / 235 |
| 4×512 稳态 / HWM | **213 / 232** | **195 / 348** |
| compare 全流程 | 213 / 258 | 195 / 348 |

两边都进 512 MB。加载和 HWM 更轻的是 burn；4×512 稳态 RSS burn 大约 +18 MB。

## 512 模型里还剩什么（breakdown / ~185 ms）

| 块 | ms | 约占 |
|---|---:|---:|
| 12× D=32 flash | **78** | **~42%** |
| MMI 72（隔离；整网 AMX 更轻） | 21–54 | ~15–25% |
| fused LN ×25 | 22 | ~12% |
| fused GELU ×12 | 17 | ~9% |
| DQL ×48 | **4.8** | 已不是大头 |
| dequant | 6 | ~3% |

下一刀 Q-block flash 已做（见下节）。

---

# Q-block D=32 flash

> 日期：2026-09-04。同一台 4 核 Xeon。
> 栈：`vendor/burn-flash-qblock` `4d580bc`（叠在 SIMD DQL 上）。
> 实现：`notes/flex-flash-qblock.md`。
> 两个进程分开跑。不要用 Mac Python 4.3 / 201。

TILE 仍是 64。Query 按 Br=16 分块，K-tile 只转一次，QK/PV 一次走 4 行。
C-lite 仍不挂钩。

mean cos **0.9950**（min 0.9876），ranking 2/2。

| 场景 | SIMD DQL | **Q-block** | Rust ort（本轮） | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok `forward_raw` | 10.1 | **10.1** | **3.7** | **2.7×** |
| packed batch `embed_passages` | 1905 | **1686** | **1168** | **1.4×** |
| 512 `forward_raw` | 185 | **149** | **53.5** | **2.8×** |
| 512 `embed_passages` | 718 | **628** | 53.5 | 含 ~476 ms SP |
| 4×512 `mem_stress` | 2827 | **2938** | **498** | **5.9×** |

隔离 flash ×12：**78 → 40 ms**。`forward_raw` **185 → 149**（−36 ms）。
`mem_stress -- 5 2048`：RSS **213 / 232 MB**。4×512 墙钟没掉（B=4 噪声）。

---

# 本机再对 Rust ort（Q-block 之后）

> 日期：2026-09-04。同一台 4 核 Xeon。**两个进程分开跑**。
> burn：`compare_ort` / `mem_stress -- 5 2048` / `breakdown`（`vendor/burn-flash-qblock` `4d580bc`）。
> Rust ort：`ort-mem -- -- 5 2048`，arena off，4 intra-op，预编码 ids。

## 延迟（公平 = 预编码 ids / 只有模型）

| 场景 | burn | **Rust ort arena off** | 倍数 | 口径 |
|---|---:|---:|---:|---|
| 16 tok `forward_raw` | **10.1 ms** | **3.7 ms** | **2.7×** | 只有模型 |
| 16 tok `embed_passages` | 13.9 ms | 3.7 ms | 3.8× | burn 含 3.5 ms SP |
| packed batch | 1686 ms | **1168 ms** | 1.4× | burn 含 SP |
| 512 `forward_raw` | **149 ms** | **53.5 ms** | **2.8×** | 只有模型 |
| 512 `embed_passages` | 628 ms | 53.5 ms | 11.7× | burn 含 ~476 ms SP |
| 4×512 `mem_stress` | 2938 ms | **498 ms** | **5.9×** | dummy ids，无 SP |

到 2× 按 ~54 ms → **~107 ms**，还差 ~42 ms。

## 吞吐（同一组活数）

| 场景 | burn q/s | burn tok/s | Rust ort q/s | Rust ort tok/s |
|---|---:|---:|---:|---:|
| 16 tok 模型 | **99** | **1584** | **274** | **4382** |
| packed batch | 4.7 | 355 | 6.0 | 508 |
| 512 模型 | **6.7** | **3436** | **18.7** | **9571** |
| 4×512 | 1.4 | 697 | 8.0 | 4112 |

## 数值

| | burn vs Python ref | Rust ort vs 同一份 ref |
|---|---:|---:|
| mean cos | **0.9950** | **0.9967** |
| min cos | 0.9876 | 0.9956 |
| ranking top-3 | 2/2 | — |

## 内存（整进程 VmRSS / VmHWM）

| 阶段 | burn Q-block | **Rust ort arena off** |
|---|---:|---:|
| 启动 | 3.2 / 3.3 | 7.4 / 7.4 |
| 模型加载后 | **87.5 / 97.4** | 155 / 235 |
| 4×512 稳态 / HWM | **213 / 232** | **195 / 348** |
| compare 全流程 | 221 / 258 | 195 / 348 |

两边都进 512 MB。burn 加载更轻、HWM 更低；稳态 RSS 大约 +18 MB。

## 512 模型里还剩什么（breakdown / ~149 ms）

| 块 | ms | 约占 |
|---|---:|---:|
| 12× D=32 flash | **40** | **~27%** |
| MMI 72（隔离；整网 AMX 更轻） | 21–54 | ~20–30% |
| fused LN ×25 | 22 | ~15% |
| fused GELU ×12 | 17 | ~11% |
| DQL ×48 | 4.9 | ~3% |
| dequant | 6 | ~4% |

下一刀 AMX packed-B 已做（见下节）。
不要再调 TILE / 不要挂钩 C-lite / 不要再融单个 DQL codegen。

---

# AMX packed-B 缓存

> 日期：2026-09-04。同一台 4 核 Xeon。
> 栈：`vendor/burn-amx-pack` `2a05a84`（叠在 Q-block flash 上）。
> 实现：`notes/flex-amx-pack.md`。
> 两个进程分开跑。不要用 Mac Python 4.3 / 201。

i8 权重缓存 AMX B 布局 + 列和。pack SSE2，zp/sums AVX-512。`tdpbusd` 串行。

mean cos **0.9950**（min 0.9876），ranking 2/2。

| 场景 | Q-block | **AMX pack** | Rust ort（本轮） | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok `forward_raw` | 10.1 | **3.2** | **3.6** | **0.9×** |
| packed batch `embed_passages` | 1686 | **1501** | **1167** | **1.3×** |
| 512 `forward_raw` | 149 | **123.6** | **53.5** | **2.3×** |
| 512 `embed_passages` | 628 | **604** | 53.5 | 含 ~478 ms SP |
| 4×512 `mem_stress` | 2938 | **2609** | **476** | **5.5×** |

`forward_raw` **149 → 123.6**（−25 ms）。短 **10.1 → 3.2**（pack 不再每 token 付）。
隔离 FFN1 仍 27.5 ms（每次新 B）。`mem_stress` RSS **233 / 257 MB**。

---

# 本机再对 Rust ort（AMX pack 之后）

> 日期：2026-09-04。同一台 4 核 Xeon。**两个进程分开跑**。
> burn：`compare_ort` / `mem_stress -- 5 2048` / `breakdown`（`vendor/burn-amx-pack` `2a05a84`）。
> Rust ort：`ort-mem -- -- 5 2048`，arena off，4 intra-op，预编码 ids。

## 延迟（公平 = 预编码 ids / 只有模型）

| 场景 | burn | **Rust ort arena off** | 倍数 | 口径 |
|---|---:|---:|---:|---|
| 16 tok `forward_raw` | **3.2 ms** | **3.6 ms** | **0.9×** | 只有模型 |
| 16 tok `embed_passages` | 6.8 ms | 3.6 ms | 1.9× | burn 含 3.5 ms SP |
| packed batch | 1501 ms | **1167 ms** | 1.3× | burn 含 SP |
| 512 `forward_raw` | **123.6 ms** | **53.5 ms** | **2.3×** | 只有模型 |
| 512 `embed_passages` | 604 ms | 53.5 ms | 11.3× | burn 含 ~478 ms SP |
| 4×512 `mem_stress` | 2609 ms | **476 ms** | **5.5×** | dummy ids，无 SP |

到 2× 按 ~54 ms → **~107 ms**，还差 ~16 ms。

## 吞吐（同一组活数）

| 场景 | burn q/s | burn tok/s | Rust ort q/s | Rust ort tok/s |
|---|---:|---:|---:|---:|
| 16 tok 模型 | **312** | **5000** | **278** | **4440** |
| packed batch | 5.3 | 398 | 6.0 | 508 |
| 512 模型 | **8.1** | **4142** | **18.7** | **9576** |
| 4×512 | 1.5 | 786 | 8.4 | 4303 |

## 数值

| | burn vs Python ref | Rust ort vs 同一份 ref |
|---|---:|---:|
| mean cos | **0.9950** | **0.9968** |
| min cos | 0.9876 | 0.9956 |
| ranking top-3 | 2/2 | — |

## 内存（整进程 VmRSS / VmHWM）

| 阶段 | burn AMX pack | **Rust ort arena off** |
|---|---:|---:|
| 启动 | 3.3 / 3.3 | 7.1 / 7.1 |
| 模型加载后 | **87.7 / 97.5** | 155 / 235 |
| 4×512 稳态 / HWM | **233 / 257** | **195 / 348** |
| compare 全流程 | 244 / 281 | 195 / 348 |

两边都进 512 MB。packed-B 缓存大约 +20 MB RSS。

## 512 模型里还剩什么（breakdown / ~124 ms）

| 块 | ms | 约占 |
|---|---:|---:|
| 12× D=32 flash | **42** | **~34%** |
| MMI 72（隔离；整网缓存更轻） | 18–50 | ~20–30% |
| fused LN ×25 | 22 | ~18% |
| fused GELU ×12 | 18 | ~15% |
| DQL ×48 | 4.9 | ~4% |
| dequant | 6 | ~5% |

下一刀 AVX-512 LN 已做（见下节）。
不要再调 TILE / 不要挂钩 C-lite / 不要再融单个 DQL codegen。

---

# AVX-512 last-axis LayerNorm

> 日期：2026-09-04。同一台 4 核 Xeon。
> 栈：`vendor/burn-simd-ln` `fbe1288`（叠在 AMX packed-B 上）。
> 实现：`notes/flex-simd-ln.md`。
> 两个进程分开跑。不要用 Mac Python 4.3 / 201。

D=384 走 4 行 AVX-512；unique 入口原地写。

mean cos **0.9952**（min 0.9903）。ranking 1/2（第二条 2/3 互换，top-1 仍中）。

| 场景 | AMX pack | **AVX-512 LN** | Rust ort（本轮） | 倍数 |
|---|---:|---:|---:|---:|
| 16 tok `forward_raw` | 3.2 | **2.5** | **2.4** | **1.0×** |
| packed batch `embed_passages` | 1501 | **1366** | **1096** | **1.2×** |
| 512 `forward_raw` | 123.6 | **103.8** | **52.4** | **2.0×** |
| 512 `embed_passages` | 604 | **591** | 52.4 | 含 ~484 ms SP |
| 4×512 `mem_stress` | 2609 | **2444** | **458** | **5.3×** |

`forward_raw` **123.6 → 103.8**（−20 ms）。隔离 LN ×25 **21.8 → 1.6 ms**。
`mem_stress` RSS **234 / 257 MB**。

---

# 本机再对 Rust ort（AVX-512 LN 之后）

> 日期：2026-09-04。同一台 4 核 Xeon。**两个进程分开跑**。
> burn：`compare_ort` / `mem_stress -- 5 2048` / `breakdown`（`vendor/burn-simd-ln` `fbe1288`）。
> Rust ort：`ort-mem -- -- 5 2048`，arena off，4 intra-op，预编码 ids。

## 延迟（公平 = 预编码 ids / 只有模型）

| 场景 | burn | **Rust ort arena off** | 倍数 | 口径 |
|---|---:|---:|---:|---|
| 16 tok `forward_raw` | **2.5 ms** | **2.4 ms** | **1.0×** | 只有模型 |
| 16 tok `embed_passages` | 6.2 ms | 2.4 ms | 2.6× | burn 含 3.5 ms SP |
| packed batch | 1366 ms | **1096 ms** | 1.2× | burn 含 SP |
| 512 `forward_raw` | **103.8 ms** | **52.4 ms** | **2.0×** | 只有模型 |
| 512 `embed_passages` | 591 ms | 52.4 ms | 11.3× | burn 含 ~484 ms SP |
| 4×512 `mem_stress` | 2444 ms | **458 ms** | **5.3×** | dummy ids，无 SP |

512 模型口径到了 **2.0×**（目标线 ~105 ms）。

## 吞吐（同一组活数）

| 场景 | burn q/s | burn tok/s | Rust ort q/s | Rust ort tok/s |
|---|---:|---:|---:|---:|
| 16 tok 模型 | **400** | **6400** | **413** | **6602** |
| packed batch | 5.9 | 438 | 6.4 | 541 |
| 512 模型 | **9.6** | **4933** | **19.1** | **9770** |
| 4×512 | 1.6 | 838 | 8.7 | 4476 |

## 数值

| | burn vs Python ref | Rust ort vs 同一份 ref |
|---|---:|---:|
| mean cos | **0.9952** | **0.9968** |
| min cos | 0.9903 | 0.9956 |
| ranking top-3 | 1/2 | — |

## 内存（整进程 VmRSS / VmHWM）

| 阶段 | burn AVX-512 LN | **Rust ort arena off** |
|---|---:|---:|
| 启动 | 3.1 / 3.1 | 7.3 / 7.3 |
| 模型加载后 | **87.6 / 97.5** | 156 / 235 |
| 4×512 稳态 / HWM | **234 / 257** | **196 / 350** |
| compare 全流程 | 246 / 288 | 196 / 350 |

两边都进 512 MB。LN 原地写没有再涨 RSS。

## 512 模型里还剩什么（breakdown / ~104 ms）

| 块 | ms | 约占 |
|---|---:|---:|
| 12× D=32 flash | **41** | **~39%** |
| MMI 72（隔离；整网缓存更轻） | 17–47 | ~25–35% |
| fused GELU ×12 | 18 | ~17% |
| fused LN ×25 | **1.6** | ~2% |
| DQL ×48 | 4.8 | ~5% |
| dequant | 6.5 | ~6% |

下一刀整层 FFN 融合已做（见下节）。
不要再调 TILE / 不要挂钩 C-lite / 不要再融单个 DQL codegen。

---

# 整层 FFN 反量化融合

> 日期：2026-09-04。同一台 4 核 Xeon。
> 栈：`vendor/burn-fuse-ffn` `5437737` + `vendor/burn-onnx-fuse-ffn` `b3353d1`
> （叠在 AVX-512 LN 上）。
> 实现：`notes/flex-fuse-ffn.md`。
> 两个进程分开跑。不要用 Mac Python 4.3 / 201。

`Cast(i32→f32) → Mul(scale) → Add(bias) [→ Gelu]` 收成一趟 AVX-512。
不融 residual。不为 DQL 重算两遍 GELU。

对拍数字待本轮 `compare_ort` / `breakdown` / `mem_stress` / `ort-mem` 补。
