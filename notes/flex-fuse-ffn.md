# 整层 FFN 反量化融合（Cast×scale+bias[+GELU]）

> 2026-09-04。改动在 `vendor/burn-fuse-ffn`（`5437737`）和
> `vendor/burn-onnx-fuse-ffn`（`1b776e9`），叠在 AVX-512 LN 上。
> 对拍数字见 `notes/poc-results.md`「整层 FFN 融合」。
> 不改 TILE。不挂钩 C-lite。不融单个 DQL。不融 residual Add。
> 不为 DQL 重算两遍 GELU。

AVX-512 LN 之后 512 `forward_raw` **103.8 ms / 2.0×** 本机 Rust ort 52.4 ms。
breakdown：fused GELU ×12 **18 ms**，dequant **6.5 ms**。生成图每层 FFN：

```
LN → DQL → MMI(FFN1) → Cast(i32→f32) → *scale → +bias → gelu
   → DQL → MMI(FFN2) → Cast → *scale → +bias → +residual → LN
```

`scale` 是 `dql_scale * weight_scale`（运行时标量，常 Unsqueeze）。
FFN1 12 处带 GELU；FFN2 / QKV / out 约 60 处只有 Cast→Mul→Add。

## 做了什么

1. **onnx-ir `DequantAffine`**  
   锚点 `Cast`。匹配 `Cast(int→float, 单消费者) → Mul(单消费者) → Add`。
   Add 的消费者里只要有 Gelu 就 `apply_gelu=true` 并替换 Gelu
   （`coalesce_gelu` 留下的 erf 残节点会占着 Add，不能要求单消费者），
   否则替换 Add。
   scale/bias 不必是常量。PHASE 4b 在 `coalesce_gelu` / `coalesce_layer_norm`
   **之后**跑，FFN1 才能看到 Gelu 节点。不融 residual（第二消费者 / 不是 last-axis affine）。

2. **flex AVX-512 kernel**  
   `y = i32 as f32 * scale + bias`，可选同一趟 GELU。
   快路径：contiguous I32、标量 F32 scale、last-axis 或标量 bias。
   `_mm512_cvtepi32_ps` + `_mm512_fmadd_ps` + 复用 `gelu_ps_avx512`。
   大 buffer 走 rayon。其它形状 fallback 到 `int_into_float + mul + add [+ gelu]`。
   GELU 路径开 FTZ/DAZ（与现有 GELU kernel 相同）。DQL 不开 FTZ。

3. **Tensor API**  
   `Tensor<D, Int>::dequant_affine` / `dequant_affine_gelu`。
   `IntTensorOps` 默认实现走 split ops；flex override 走融合 kernel。

`unsafe` 只在 burn-flex。burn-onnx 无 `unsafe`。

## 单测

- flex `--release --lib -- dequant_affine`：6/6
- onnx-ir `--lib -- coalesce_dequant`：5/5
- burn-onnx `--lib -- dequant_affine`：2/2 codegen 快照

## 对拍（本机 4 核 Xeon，flex release；进程分开跑）

待 `compare_ort` / `breakdown` / `mem_stress` / `ort-mem` 跑完后补。

合理预期 512 `forward_raw` 再掉 **10–20 ms**（104 → ~85–95）。
不要指望到 1×。flash 仍是最大头。
