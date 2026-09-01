# E5-small int8 导入验证

## 里程碑

**2026-09-01**：burn-onnx 成功导入 `intfloat/multilingual-e5-small`（int8 ONNX，118MB，1630 节点），并跑通 forward。

```
Loading model from .../model_qint8_avx512_vnni.bpk...
Model loaded.
Running forward pass...
last_hidden_state shape: Shape { dims: [1, 8, 384] }
Done.
```

## 额外修复：`to_i64_vec` 支持小整数类型

e5 模型里有个 Constant 节点的值是 U8，codegen 试图转 I64 时 panic：
```
UnsupportedConversion { from: U8, to: I64 }
```

修复 `burn-onnx/crates/onnx-ir/src/ir/tensor_data_ext.rs:90`：
```rust
DType::I8 | DType::I16 | DType::U8 | DType::U16 => self.try_to_vec_as::<i64>(),
```

## DQL broadcast 修复

初版 DQL codegen 假设输入是 1 维，e5 是 3 维 `[batch, seq, hidden]`，导致：
```
expected `Tensor<3>`, found `Tensor<1>`
```

修复：根据输入 rank 动态 unsqueeze scale/zero_point：
```rust
let unsqueeze_dims: Vec<isize> = (0..x_rank as isize).collect();
let y_scale_broadcast = (#y_scale).clone().unsqueeze_dims(&[#(#unsqueeze_dims),*]);
```

## 模型结构

- 输入：`input_ids`, `attention_mask`, `token_type_ids`（均 `Tensor<2, Int>`）
- 输出：`Tensor<3>`（`last_hidden_state` [batch, seq, 384]）
- 内部被 partition 成 13 个 submodule（节点数 > 200 自动拆分）
- 权重文件：`model_qint8_avx512_vnni.bpk`（118MB）

## 下一步

阶段 3 已完成（`notes/poc-results.md`）。下一跳是阶段 4（`notes/stage-4.md`）：cubek i8 GEMM。
