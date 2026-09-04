# 收成 GELU + LayerNorm

> 2026-09-03。改动在 `vendor/burn-onnx-coalesce-gelu-ln`（`68153cc`）和
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
   平方认 `Pow(...,2)` 或 `Mul(c,c)`。epsilon 下界用 `1e-13`（e5 的 f32 `1e-12`
   转成 f64 是 ≈9.999e-13，闭区间 `[1e-12,…]` 会拒掉）。γ/β 必须是 rank-1 且 `value()` 有数据
   （否则 `LayerNorm::field()` 的 `static_shape_known()` 会炸，或变成全零权重）。
   最后那个 Add 换成 `LayerNormalization`，`axis=-1`，`stash_type=0`
   （跳过 codegen 多出来的 f32 cast）。
   PHASE 4b 插入；**不**再跑 `LayerNormProcessor::infer_types`（opset 17）。
   现成 `nn::LayerNorm` → flex SIMD `layer_norm`。

3. **flex `gelu` rayon**  
   融合之后仍是一趟标量 `libm::erff`。`[1,512,1536]=786432 > PARALLEL_THRESHOLD`
   （256K）时 `par_chunks_mut`（16K）。**不**换 A&S 近似，以免再漂 cos。

`unsafe` 没有进 burn-e5q / burn-onnx。

## 验收（编译期，已过）

生成图：`activation::gelu` **12**，`.erf(` **0**，`LayerNormConfig::new(384)` / `*_ln.forward` **25**，`mean_dim` **0**。
`module::attention` 仍是 12，`matmul_integer` 仍是 72。

副作用：γ/β 改走 `collect_tensors` 之后，原先的 `self.constantN.val()` 变成 50 条 unused warning，不影响正确性。

## 单测

- onnx-ir `coalesce_gelu` 6/6（e5 Div、Mul 1/√2、half-on-x、错误 scale、Erf 另有消费者、Unsqueeze 常量）
- onnx-ir `coalesce_layer_norm` 5/5（e5 Pow、Mul 平方、错误 axis、Sqrt 另有消费者、`f32_1e12_eps_still_matches`）
- flex `test_gelu_small_matches_libm` / `test_gelu_parallel_matches_libm`

## 对拍（本机 4 核 Xeon，flex release）

| 口径 | 融合前 | **融 GELU/LN** | 差 |
|---|---:|---:|---:|
| mean cos | 0.9946 | **0.9950**（min 0.9886） | 不漂 |
| 16 tok `embed_passages` | 28.4 ms | **28.6 ms** | 噪声 |
| 512 `forward_raw` | **639 ms** | **636 ms** | **≈ 0** |
| 512 `embed_passages` | 1099 ms | **1151 ms** | 含 SP，勿比 |

隔离（min of repeats，`breakdown`）：

| 块 | 展开 | 融合 | 差 |
|---|---:|---:|---:|
| GELU ×12 | 116.8 | **83.3** | −33.5（4 核 rayon，仍是 `libm::erff`） |
| LN ×25 | 44.2 | **21.8** | −22.4 |
| 合计 | 161 | 105 | **~56 ms** |

生成代码确实调 `activation::gelu` 和 `nn::LayerNorm.forward`。隔离最多能省 56 ms，预期「639 → 480–520」**落空**：整网 `forward_raw` 几乎没动。
`breakdown` 的 isolated sum 仍按展开 LN+GELU 记账，所以 632 vs 636 对齐不能用来证明图没走融合。

候选（未再追）：图上 GELU 输入未必 unique/contiguous，走分配路径；`LayerNorm.forward` 比直接 `module::layer_norm` 多 `.val()` / dtype 分支。50 个死 `constantN.val()` 解释不了 50 ms。

## 这一刀的结论

codegen **做成了**（12 GELU、25 LN、0 erf、cos 不漂）。延迟 **没按预期掉**。
到 2× 仍然主要靠整数 flash 和更快的 MMI。不要再用 A&S 换默认 erf，不要再融单个 DQL。
