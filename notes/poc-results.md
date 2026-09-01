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

## Linux 复测（2026-09-01）

同一套 flex `compare_ort` 在 4 核 Xeon（`avx512_vnni` + `amx_int8`）上重跑：短文本 130 ms、512 tok 3.82 s，相对 `ref_data.json` 的 Mac ort 仍是 **19–30×**。本机 Python ort 1.29 把 512 tok 打到 50 ms，倍数变成 **~77×**——ort 吃到了 VNNI/AMX，flex 没有。完整分析见 `notes/verify-burn-vs-ort.md`。

## 下一步（按优先级）

1. **阶段 4**（`notes/stage-4.md`）：在 [TsaoLun/cubek](https://github.com/TsaoLun/cubek) 做 i8 GEMM，经本仓库 `[patch]` 接入 `e5-embed --features cpu`。
2. 内存：若阶段 4 后 scratch 仍超预算，默认预算降 2048（已验证可行）。
3. 上线前用真实语料做 recall@k 评估（替代绝对 cos 阈值）。
