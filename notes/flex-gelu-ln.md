# 收成 GELU + LayerNorm

> 2026-09-03。改动在 `vendor/burn-onnx-coalesce-gelu-ln`（`c0f9a6d`）和
> `vendor/burn-flex-par-gelu`（`319336c`）。对拍数字见 `notes/poc-results.md`「融 GELU/LN」。

e5（opset 11）没有 `Gelu` / `LayerNormalization` 节点。生成图把它们展开成
5 趟 erf 路径和 8～9 趟 LN。隔离计时：12× GELU **117 ms**，25× LN **44 ms**
（占模型 639 ms 的 25%）。

## 做了什么

1. **`onnx-ir` `coalesce_gelu`**  
   锚点 `Erf`。匹配
   `(Div(x,√2) | Mul(x,1/√2)) → Erf → Add(1) → Mul(x) → Mul(0.5)`，
   也认 `0.5*x` 先乘再乘 `(1+erf)`。常量可剥 Unsqueeze/Squeeze/Identity/Reshape。
   最后那个 Mul 换成 `NodeType::Gelu`，输出名不变。
   PHASE 4b 插入；**不**再跑 `GeluProcessor::infer_types`（opset 20）。
   codegen 已有 `burn::tensor::activation::gelu`。

2. **`onnx-ir` `coalesce_layer_norm`**  
   锚点 `Sqrt`。匹配最后一维 `keepdims=1` 的
   `mean → (x-mean) → square → mean → +eps → sqrt → div → *γ → +β`。
   平方认 `Pow(...,2)` 或 `Mul(c,c)`。γ/β 必须是 rank-1 且 `value()` 有数据
   （否则 `LayerNorm::field()` 的 `static_shape_known()` 会炸，或变成全零权重）。
   最后那个 Add 换成 `LayerNormalization`，`axis=-1`，`stash_type=0`
   （跳过 codegen 多出来的 f32 cast）。
   PHASE 4b 插入；**不**再跑 `LayerNormProcessor::infer_types`（opset 17）。
   现成 `nn::LayerNorm` → flex SIMD `layer_norm`。

3. **flex `gelu` rayon**  
   融合之后仍是一趟标量 `libm::erff`。`[1,512,1536]=786432 > PARALLEL_THRESHOLD`
   （256K）时 `par_chunks_mut`（16K）。**不**换 A&S 近似，以免再漂 cos。

`unsafe` 没有进 burn-e5q / burn-onnx。

## 验收（编译期）

```
grep activation::gelu     generated.rs   # = 12
grep LayerNorm            generated.rs   # ≈ 25 个 field + forward
grep '.erf('              generated.rs   # = 0
grep mean_dim             generated.rs   # 大降（只剩非 LN 的 reduce）
```

少一层就回去查 matcher，不要先跑 `compare_ort`。

## 单测

- onnx-ir `coalesce_gelu` 6/6（e5 Div、Mul 1/√2、half-on-x、错误 scale、Erf 另有消费者、Unsqueeze 常量）
- onnx-ir `coalesce_layer_norm` 4/4（e5 Pow、Mul 平方、错误 axis、Sqrt 另有消费者）
- flex `test_gelu_small_matches_libm` / `test_gelu_parallel_matches_libm`

## 预期

只收 GELU/LN：模型 639 → **约 480–520 ms**（~9× Rust ort 53.8 ms）。
cos 仍约 0.99x。到 2× 还要整数 flash 和更快的 MMI，不在这一刀。
