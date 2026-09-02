# burn-e5q: 用 burn 生态替换 inmotion-social 的 ort

> 把 `intfloat/multilingual-e5-small`（int8 ONNX）的在线推理从 ONNX Runtime 迁到 burn/burn-onnx/cubek。
> 目标：**边学、边做、边贡献**。本仓库是 PoC 与编排；codegen / kernel 分别落在 fork，经 git `rev` 钉住（见 `AGENTS.md`）。
>
> | 仓库 | 角色 |
> |---|---|
> | 本仓库 | tokenizer、对拍、内存压测、阶段文档 |
> | [TsaoLun/burn-onnx](https://github.com/TsaoLun/burn-onnx) `add-dynamic-quantize-linear` | DQL + clone-tracking 修复 |
> | [TsaoLun/cubek](https://github.com/TsaoLun/cubek) | 阶段 4 i8 GEMM（实现已完成；`[patch]` 覆盖 burn 对 cubek 的依赖） |
> | burn / cubecl / burn-onnx | 阶段 4 接线与 host-native JIT；`cursor[bot]` 无法写 TsaoLun fork，暂钉 `TsaoLun/burn-e5q` 的 `vendor/*` snapshot（见 `AGENTS.md`） |

云端 agent 请先读 **`AGENTS.md`**，阶段 4 操作手册是 **`notes/stage-4.md`**。

## 现状速查

| 项 | 现状 | 证据 |
|---|---|---|
| 线上模型 | `model_qint8_avx512_vnni.onnx`，118MB，1630 节点 | `inmotion-social/data/models/multilingual-e5-small/` |
| burn-onnx 缺的唯一算子 | ~~`DynamicQuantizeLinear`~~ 已在本地实现（onnx-ir processor + codegen） | `crates/onnx-ir/src/node/dynamic_quantize_linear.rs`、`crates/burn-onnx/src/import/burn/node/dynamic_quantize_linear.rs` |
| MatMulInteger codegen | 已存在，走真 I32 整数 matmul | `burn-onnx/crates/burn-onnx/src/import/burn/node/matmul_integer.rs:20-100` |
| burn 后端 int matmul | flex: naive i32 loop；cubecl: 通用 int matmul，**无 VNNI** | `burn/crates/burn-flex/src/ops/matmul.rs:521`、`burn/crates/burn-cubecl/src/ops/int_tensor.rs:86` |
| inmotion-social 推理 | ort 2.0.0-rc.9 + vendored ORT 1.20，CPU EP，arena 关闭 | `inmotion-social/src/rec/text_embed.rs:622-669` |
| 容器限制 | 512MB，Dockerfile 设 `MALLOC_ARENA_MAX=2` | `inmotion-social/Dockerfile:77` |

**一句话：burn-onnx 加 DQL 只解锁「图能导入」，替换 ort 还要解决性能与内存。**

---

## 阶段 0：学习 + 环境对齐（1–2 天）

在写代码前把三件事跑通。

1. 读 burn-onnx 的算子开发套路：
   - `burn-onnx/AGENTS.md`、`DEVELOPMENT-GUIDE.md`、`crates/onnx-ir/src/node/quantize_linear.rs`
   - `burn-onnx/crates/burn-onnx/src/import/burn/node/quantize_linear.rs`（上面刚看过的实现）
2. 跑通最小量化模型导入：
   - 在 `burn-e5q` 新建 `crates/dql-poc/`，`build.rs` 用 `ModelGen::new().input("model.onnx")` 导入一个只有 `DynamicQuantizeLinear + MatMulInteger` 的 2 节点模型
   - 后端先用 `burn-flex`（纯 Rust，无 JIT），确认能编译
3. 对齐版本：确认 inmotion-social 的 `burn v0.22.0-pre.2` 与 burn-onnx `0.22.0-pre.3` 的兼容关系；必要时在 inmotion-social 的 Cargo.toml 里升到同一 pre 版本。

**产出**：`burn-e5q/notes/onnx2burn-pipeline.md`，记录 ModelGen 入口、生成文件位置、`.bpk` 权重结构。

---

## 阶段 1：burn-onnx 实现 `DynamicQuantizeLinear`（3–5 天）

**目标**：让 `onnx2burn` 不再报 `Unsupported ONNX operation(s): DynamicQuantizeLinear`。

### 1.1 在 `onnx-ir` 里加 node processor
文件：`burn-onnx/crates/onnx-ir/src/node/dynamic_quantize_linear.rs`（新建）

要点：
- 实现 `NodeProcessor`，注册到 `registry.rs`
- 从 `unsupported.rs` 的 `define_placeholder_node!` 里删掉 `DynamicQuantizeLinearNode`
- 输入：1 个张量；输出：3 个（y, y_scale, y_zero_point）
- 类型推断：y 与输入同 shape，dtype 由 attribute 或默认 u8/i8；scale 是标量 f32；zero_point 是标量同 dtype

参照：`quantize_linear.rs`、`dequantize_linear.rs`、`matmulinteger.rs` 的 `NodeProcessor` 结构。

### 1.2 在 `burn-onnx` 里加 codegen
文件：`burn-onnx/crates/burn-onnx/src/import/burn/node/dynamic_quantize_linear.rs`（新建）

公式（ONNX 语义）：
```
y_scale = (max(x) - min(x)) / (qmax - qmin)
y_zero_point = clamp(round((0 - min(x)) / y_scale), qmin, qmax)
y = clamp(round(x / y_scale) + y_zero_point, qmin, qmax)
```

用 burn tensor API 写：
- `min/max`：`tensor.clone().min_dim(dim)` / `max_dim(dim)`
- `round/clamp/cast`：与 `quantize_linear.rs` 同一套
- 标量输出：可用 `into_scalar()` 或保持 0-rank Tensor

注意 DQL 是**逐张量动态量化**，不是逐通道；先只支持 per-tensor，axis 相关逻辑留给后续。

### 1.3 验证
- 官方 ONNX 测试：`crates/onnx-official-tests/expectations.toml` 里 6 条 `test_dynamicquantizelinear*` 应从 `skip-codegen` 变为 `pass`
- 本地 `cargo test -p burn-onnx dynamic_quantize`
- 跑 `cargo xtask retriage` 刷新状态

**产出**：向 burn-onnx 提交 PR「Add DynamicQuantizeLinear support」。这是最小、最干净的贡献。

---

## 阶段 2：multilingual-e5-small model-check（2–3 天）

**目标**：在 burn-onnx 里建立 e5-small 的回归测试，证明「导入 + 数值正确」。

1. 参照 `crates/model-checks/all-minilm-l6-v2/` 新建 `crates/model-checks/multilingual-e5-small/`
2. 模型来源：Xenova 的 `multilingual-e5-small` ONNX（opset 16，无 bool And，避开 burn#4771），或直接用项目里的 `model_qint8_avx512_vnni.onnx`
3. 输入构造：`input_ids [B, L]`、`attention_mask [B, L]`，可选 `token_type_ids`
4. 输出验证：与 ort 输出的 `last_hidden_state` 做 cosine similarity / max abs diff 对拍
5. 后端覆盖：`flex` + `cpu`（cubecl），动态 batch 1/4/16 × seq 32/128/512

**产出**：burn-onnx PR「Add multilingual-e5-small model check」。同时得到一份「这个图到底能不能跑」的实测报告。

---

## 阶段 3：inmotion-social 集成 PoC（3–4 天）

**目标**：在 `burn-e5q` 里先跑通，再决定是否合并回 inmotion-social。

### 3.1 建实验 crate
`burn-e5q/crates/e5-embed/`
- `build.rs`：ModelGen 导入 e5 int8 ONNX
- `src/lib.rs`：暴露 `embed(texts: &[&str]) -> Vec<Vec<f32>>`
- 复用 inmotion-social 的 sentencepiece tokenizer、E5 前缀、mean pooling、L2 normalize

### 3.2 与 ort 对拍
写 `bins/compare_ort.rs`：
- 同一批句子，分别走 ort 和 burn
- 比较 384 维向量的 cosine similarity（目标 > 0.999）
- 统计延迟：单条、batch 8/32/128

### 3.3 内存压测
模拟 inmotion-social 的 4096 token 预算，记录 RSS 峰值：
- 512MB 容器内跑 `cargo run --release`
- 关注 CubeCL JIT 预热是否导致初始 spike

**产出**：`burn-e5q/notes/poc-results.md`，含数值误差、延迟对比、内存曲线。只有这里达标后才进阶段 4。

---

## 阶段 4：cubek i8 GEMM（性能攻坚）

**前置已满足**：阶段 3 测得 flex 比 ort VNNI 慢 19–45×。手册：`notes/stage-4.md`。
实现说明：`notes/stage-4-impl.md`。`cursor[bot]` 无法写 TsaoLun fork，代码以 orphan snapshot 钉在本仓库 `vendor/*`。

### 4.1 cubek 整数 GEMM — 已实现（`vendor/cubek-add-i8-gemm`）
- `U8/I8 × U8/I8 → I32` 走 tiled CpuGemm；K 面板按输入字节计，优先 `tile_k` 为 4 的倍数
- 叶子仍是 tiled `SUM_PROD`；host-native LLVM TM（`vendor/cubecl-host-native-jit`）让 O3 有机会 autovec 到 VNNI
- **不要**套 cubek-quant 的 per-block `mm_scaled`（那是 Q4S/Q8S，不是 ONNX MatMulInteger）

### 4.2 burn-cpu 接线 — 已实现（`vendor/burn-route-int8-matmul`）
- `CubeBackend::int_matmul`：两边都是 I8/U8 时输出 I32，强制 `MatmulStrategy::CpuGemm`
- flex：混合 u8/i8 先 `int_cast(I32)` 再走现有 i32 GEMM

### 4.3 burn-onnx codegen — 已实现（`vendor/burn-onnx-keep-int8-matmul`）
- MatMulInteger **不再** `cast(I32)` 再 `.matmul()`，保留输入 dtype
- zp 用代数恒等式：`(A-za)@(B-zb) = A@B − za·sum_k(B) − sum_k(A)·zb + za·zb·K`

### 4.4 回归
`e5-embed --features cpu` 的 `compare_ort` / `mem_stress`；结果追加到 `notes/poc-results.md`。

**产出**：本仓库 pin + 对拍。真 fork 分支（`add-i8-gemm` / `route-int8-matmul` / `keep-int8-matmul` / `host-native-jit`）待有写权限后再推。

---

## 阶段 5：替换 inmotion-social 的 ort（最终集成）

1. `Cargo.toml`：去掉 `ort` / `ort-sys`，加 `burn` / `burn-onnx` 生成模型
2. `text_embed.rs`：把 `run_batch` 里的 `session.run` 换成生成的 `Model::forward`
3. `Dockerfile`：删除 `COPY data/onnxruntime` 和 `ORT_LIB_LOCATION`
4. 保留：tokenizer、PG 缓存、token 预算、E5 前缀、降级为零向量逻辑
5. 全量回归：`cargo test --features postgres`、部署到 staging 观察 24h

---

## 学习清单

- ONNX opset 与量化语义：`onnx-spec/ops/DynamicQuantizeLinear.md`
- burn-onnx 架构：`AGENTS.md`、`DEVELOPMENT-GUIDE.md`、`onnx-ir/src/phases/`
- CubeCL 编程模型：`cubecl/README.md` 的 Vector/Plane/CubeDim/CubeCount
- AVX512-VNNI：`vpdpbusd` 指令与 4 元素 dot product
- burn 后端 trait：`burn-tensor/src/bridge/ops/int.rs` 的 `IntOps`

---

## 里程碑 checklist

- [x] 阶段 0：能编译一个含 DQL 的 2 节点 burn 模型（`burn-e5q/crates/dql-poc` 跑通，数值与 ONNX 语义一致）
- [x] 阶段 1：burn-onnx 本地实现 DQL，e5-small 图不再报 DQL unsupported（fork 分支 `add-dynamic-quantize-linear`，含 graph.rs clone-tracking 修复；官方 6 条 node test 编译通过）
- [x] 阶段 2：model-check 通过（`cargo xtask model-check --model multilingual-e5-small`，last_hidden_state 与 pooled 均在 1e-4 内匹配 ort；fork 分支 `add-multilingual-e5-small-model-check`）
- [x] 阶段 3：PoC 对拍完成（`notes/poc-results.md`：tokenizer 9/9 ✓，int8 cos 0.996 属固有跨引擎分歧，top-1 检索 2/2；延迟差 19–45×；4096 预算 RSS 640MB 超标，降 2048 → 416MB ✓）
- [ ] 阶段 4：i8 GEMM 栈已接线（见 `notes/stage-4-impl.md`）；对拍延迟待记入 `notes/poc-results.md`
- [ ] 阶段 5：inmotion-social 部署纯 burn 版本
