# flex 融合 DynamicQuantizeLinear

> 2026-09-02。改动在 `vendor/burn-route-int8-matmul`（`2d1084f`）和
> `vendor/burn-onnx-keep-int8-matmul`（`7cc2d36`）。对拍数字见 `notes/poc-results.md`「融 DQL」。

## 做了什么

e5 图里有 **96 个 DQL**。原先每个展开成 ~10 趟 tensor op（`clone+min`、`clone+max`、`div→round→add→clamp→cast`），大的是 softmax 后的 `[12,S,S]`。

1. **`Tensor::dynamic_quantize_linear(self) -> (Tensor<D,Int>, Tensor<1>, Tensor<1,Int>)`**
   - `FloatTensorOps::float_dynamic_quantize_linear`
   - 默认实现：现有 min/max/round/clamp/expand/cast（cubecl 等不改也能跑）
   - flex：一遍 minmax + 一遍 ties-to-even 量化写 u8；大 tensor 走 rayon
2. **burn-onnx DQL NodeCodegen** 改成一行
   `let (y, y_scale, y_zp) = x.dynamic_quantize_linear();`
   没有重写 NodeProcessor（scale/zp 仍是 `ScalarTensor`，图边界 `into_scalar` 照旧）。

`unsafe` 没有进 burn-e5q / burn-onnx。flex 量化循环交给 LLVM autovec（`target-cpu=native`）。

## 单测

- flex `dql` 8/8（official 输入的 zp/scale、fused vs 展开、全正/全负、rank-3、e5-like `[16,384]`、全零、ties-to-even）
- burn-onnx DQL insta 2/2

f32 `/ 255` 与 numpy 先 f64 再 cast 会在 `.5` 边界差 1 个 bin；fused 与 Burn 展开路径 bit 一致，不引入第二份漂移。

## 对拍（本机 4 核 Xeon，flex）

| 场景 | VNNI+zp | **融 DQL** | vs VNNI | **vs 本机 Rust ort** |
|---|---:|---:|---:|---:|
| 16 tok | 33.2 ms | **29.6 ms** | 1.12× | **12×**（2.4 ms） |
| packed 7 条 | 8.73 s | **7.07 s** | 1.23× | **7.6×**（936 ms） |
| 512 tok | 1.46 s | **1.27 s** | 1.15× | **24×**（53.8 ms） |

mean cos **0.9960**（Rust ort 对同一份 Python ref 也是 0.9968）。加载 RSS 87.6 MB；`mem_stress -- 5 2048` 稳态 **213 / 315 MB**。

整进程 Rust ort（`ort-mem`，arena off）：加载 153 MB，4×512 **162 / 268 MB**，compare 全流程 **193 / 346 MB**。burn 大约 +51 MB RSS。表见 `notes/poc-results.md`「融 DQL」。

吞吐：短 **33.8 q/s / 540 tok/s**；512 **0.79 q/s / 403 tok/s**。

详见 `notes/poc-results.md`「融 DQL」。
