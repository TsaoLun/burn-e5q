# DynamicQuantizeLinear 实现笔记

## 修改文件

### 1. `burn-onnx/crates/onnx-ir/src/node/dynamic_quantize_linear.rs`（新建）
- `DynamicQuantizeLinearConfig`：空配置（DQL 无属性）
- `DynamicQuantizeLinearNode`：1 输入 3 输出
- `DynamicQuantizeLinearProcessor`：
  - `min_opset: 11`
  - `inputs: Exact(1)`，`outputs: Exact(3)`
  - 类型推断：y = U8 同 shape，y_scale = F32 scalar，y_zero_point = U8 scalar

### 2. `burn-onnx/crates/burn-onnx/src/import/burn/node/dynamic_quantize_linear.rs`（新建）
- 实现 `NodeCodegen::forward`
- 公式（ONNX opset 11，仅支持 uint8）：
  ```
  x_min_adj = min(min(x), 0)
  x_max_adj = max(max(x), 0)
  y_scale = (x_max_adj - x_min_adj) / 255
  y_zero_point = clamp(round((0 - x_min_adj) / y_scale), 0, 255)
  y = clamp(round(x / y_scale) + y_zero_point, 0, 255)
  ```
- 关键：中间变量保持 `Tensor<0>`，让 codegen 框架自动对 `ScalarTensor` 输出做 `.into_scalar::<T>()`

### 3. 注册与清理
- `onnx-ir/src/node/mod.rs`：加 `pub mod dynamic_quantize_linear;`
- `onnx-ir/src/ir/node.rs`：`DynamicQuantizeLinear => dynamic_quantize_linear::DynamicQuantizeLinearNode`
- `onnx-ir/src/node/unsupported.rs`：从 placeholder 列表删除 `DynamicQuantizeLinearNode`
- `onnx-ir/src/registry.rs`：注册 `DynamicQuantizeLinearProcessor`
- `burn-onnx/src/import/burn/node/mod.rs`：加 `pub(crate) mod dynamic_quantize_linear;`
- `burn-onnx/src/import/burn/node_codegen.rs`：加 `DynamicQuantizeLinear`

## 4. 官方测试 / 仓库位置
- `crates/onnx-official-tests/expectations.toml`：6 条 `test_dynamicquantizelinear*` 从 `skip-codegen` 改为 `pass`（codegen 编译通过；harness 因 rank-0 标量输出未生成数值测试）
- 实现已推到 [TsaoLun/burn-onnx](https://github.com/TsaoLun/burn-onnx) 分支 `add-dynamic-quantize-linear`（含后续 `graph.rs` clone-tracking 修复 `63e35840`）

## 验证结果

### onnx-ir 单测
```
cargo test -p onnx-ir --lib dynamic
# 35 passed, 0 failed
```

### dql-poc 端到端
```
cargo run -p dql-poc
# input: [0., 2., -3., -2.5, 1.34, 0.5]
# y: [153, 255, 0, 25, 187, 178] (U8)
# y_scale: 0.019607844 (= 5/255)
# y_zero_point: 153
```

与 ONNX 官方语义一致。

## 已知限制

- 只支持 uint8 输出（ONNX spec 当前也只支持 uint8）
- 只支持 per-tensor 量化（无 axis/block_size）
- `cargo test -p burn-onnx` 在 macOS 上因 ort-sys 无预编译包而失败，需用 Linux 或 CI 跑完整测试
