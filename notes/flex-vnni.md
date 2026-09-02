# flex VNNI + 融合 zp

> 2026-09-02。改动在 `vendor/burn-route-int8-matmul`（`30cab97`）和
> `vendor/burn-onnx-keep-int8-matmul`（`3a2bf47`）。
> 本仓库加了 `.cargo/config.toml`：`rustflags = ["-C", "target-cpu=native"]`。

## 做了什么

1. **`target-cpu=native`**：release 不再只编 SSE2。本机 `rustc --print cfg -C target-cpu=native` 能看到 `avx512vnni`。
2. **VNNI 微内核**（`burn-flex` `int_gemm_vnni.rs`）：运行时检测 `avx512f+bw+vnni`，走 `vpdpbusd`。
   e5 热路径是 **u8×i8**。i8×i8 / u8×u8 通过 ±128 映射到同一指令再补偿。无 VNNI 时回退分块三重循环（已去掉 `if av == 0`）。
3. **融合 zp**：`Tensor::matmul_integer(rhs, zp_a, zp_b)` + `IntTensorOps::int_matmul_integer`。
   默认实现仍是代数恒等式；flex 在同一趟 GEMM 里做
   `C = A@B − za·sum_b − sum_a·zb + K·za·zb`。
   burn-onnx 的 MatMulInteger codegen 改为调用该方法，不再展开一串 `sum_dim`。
   DQL 本身没融进 GEMM（图改写，且 AGENTS.md 禁止重做 DQL NodeProcessor）。

`unsafe` 只在 flex 的 `#[target_feature]` 内核里（loadu / mask store）。burn-e5q / burn-onnx 无新增 `unsafe`。

## 单测

- flex `int_gemm` 9/9（含 u8×i8 / i8×i8 / u8×u8、ragged K、标量/逐列 zp、并行 M 切分）
- flex `int_matmul` 5/5（含 fused zp vs 代数展开对照）
- burn-onnx MatMulInteger insta 11/11

## 对拍（本机 4 核 Xeon）

| 场景 | 分块 flex | VNNI+zp | vs 分块 | vs Mac ort |
|---|---:|---:|---:|---:|
| 16 tok | 75.7 ms | **33.2 ms** | 2.3× | 7.7× |
| 8 条 batch | 18.1 s | **8.73 s** | 2.1× | 6.2× |
| 512 tok | 2.80 s | **1.46 s** | 1.9× | 7.3× |

mean cos **0.9960**。加载 RSS 88 MB；`mem_stress -- 5 2048` 峰值 **215 MB**（4×512 稳态 ~6.4 s）。

512 tok 已贴近先前估的非 GEMM 下限（~1.3 s：96 个 DQL + LN）。再往下要融 DQL，不是再写 GEMM。

详见 `notes/poc-results.md`「flex VNNI」。
