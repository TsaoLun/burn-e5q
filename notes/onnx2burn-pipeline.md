# onnx2burn pipeline 学习笔记

## 环境

- workspace：`burn-e5q`（git 依赖，无本地 path）
- 实验 crate：`crates/dql-poc`、`crates/e5-embed`
- burn rev：`af844911be6efb6745301c1c2c5e695d6571b316`
- burn-onnx / onnx-ir：`https://github.com/TsaoLun/burn-onnx` @ `63e35840`（分支 `add-dynamic-quantize-linear`）
- cubek：`https://github.com/TsaoLun/cubek`，经 `[patch."https://github.com/tracel-ai/cubek"]` 覆盖 burn 的 cubek 依赖

## ModelGen 用法

```rust
ModelGen::new()
    .input("model.onnx")
    .out_dir("model/")
    .development(true)
    .run_from_script();
```

- 输出：`$OUT_DIR/model/<name>.rs` + `$OUT_DIR/model/<name>.bpk`
- `development(true)` 额外生成 `.onnx.txt` 和 `.graph.txt` 调试图

## 阶段 0 卡点（已解决）

官方 ONNX node test `test_dynamicquantizelinear/model.onnx` 曾在 **PHASE 3: Type Inference** 失败：

```
Unsupported ONNX operation(s): DynamicQuantizeLinear (node 'dynamicquantizelinear1')
```

DQL 已在 fork 实现。`dql-poc` 端到端：`y=[153,255,0,25,187,178]`，与 ONNX 语义一致。

## 官方测试模型结构

`test_dynamicquantizelinear/model.onnx`（184 bytes）：
- 输入：`x` FLOAT [?, ?]
- 节点：`DynamicQuantizeLinear`
- 输出：`y` UINT8、`y_scale` FLOAT scalar、`y_zero_point` UINT8 scalar
