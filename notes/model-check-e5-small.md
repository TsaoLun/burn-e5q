# multilingual-e5-small model-check 笔记

> 对应 PLAN 阶段 2。位置：`burn-onnx/crates/model-checks/multilingual-e5-small/`。

## 新增文件

| 文件 | 作用 |
|---|---|
| `Cargo.toml` | 依赖 burn (flex)、burn-store、tokenizers、`model-checks-common` |
| `build.rs` | `ModelGen` 导入缓存目录里的 `multilingual-e5-small_opset16.onnx`；支持 `E5_MODEL_PATH` 覆盖 |
| `get_model.py` | 从 HF 下载 Xenova FP32 ONNX → opset 11→16 升级 → shape inference → 用 onnxruntime 生成 `test_data.pt` 参考数据 |
| `src/main.rs` | 加载生成的 burn 模型 + `test_data.pt`，前向 → mean pooling → 与 ort 参考对拍（容差 1e-4） |
| `README.md` | 使用说明 |

并在 `xtask/src/model_check.rs` 注册 `ModelInfo`，可用：

```bash
cargo xtask model-check --model multilingual-e5-small all
```

## 模型与数据

- 模型源：`Xenova/multilingual-e5-small`（FP32，避开 int8 路线先做数值正确性）
- 处理后：`multilingual-e5-small_opset16.onnx`（~471MB，XLM-RoBERTa，384 维）
- 输入：`input_ids` / `attention_mask` / `token_type_ids`，均 `[1, 128]` I64
- 参考输出：`last_hidden_state [1,128,384]` + ort 侧 mean pooling 后的 `pooled_embeddings [1,384]`
- 缓存目录：`~/Library/Caches/burn-onnx/model-checks/multilingual-e5-small/`

## 验证结果（2026-09-01，flex 后端，release）

```
✓ last_hidden_state matches reference data within tolerance (1e-4)
✓ pooled_embeddings matches reference data within tolerance (1e-4)
All 1 model(s) passed
```

冷启动推理 ~160ms（128 tokens，flex 后端，未调优）。

## 踩坑记录

1. **HF 下载需代理**：`huggingface_hub` 走 `http_proxy`/`https_proxy`；**不要**设 `all_proxy=socks5://...`，uv 环境的 httpx 缺 `socksio` 会直接 ImportError。
2. **torch + numpy 2.x 不兼容**：`torch.from_numpy` 报 `_ARRAY_API not found` / `Numpy is not available`，在 script 依赖里 pin `numpy<2` 解决。
3. **opset 11 → 16**：Xenova 原始模型是 opset 11，用 `onnx.version_converter` 升级 + `onnx.shape_inference` 后才过 burn-onnx 导入。
4. **main.rs 的 forward 签名**：生成的 e5 模型 `forward(input_ids, attention_mask, token_type_ids) -> Tensor<3>`（只有 last_hidden_state 一个输出），pooled 是在 main.rs 里手动 mean pooling 算的，与 get_model.py 里的参考保持一致。

## 与 int8 路线的关系

本 model-check 用的是 **FP32** 模型，证明的是「图导入 + 数值正确」。inmotion-social 线上的 `model_qint8_avx512_vnni.onnx` 含 96 个 DQL 节点，现在 DQL 已支持，理论上也能导入，但性能取决于阶段 4 的 i8 GEMM。后续可以给 model-check 加一个 `E5_MODEL_PATH` 指向 int8 模型的变体，验证 DQL+MatMulInteger 链路的数值。
