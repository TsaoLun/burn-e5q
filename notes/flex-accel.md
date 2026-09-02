# flex 整数 GEMM 加速

> 2026-09-02。改动在 `vendor/burn-route-int8-matmul`（`005354fd`），crate `burn-flex`。
> 默认 `cargo run --release -p e5-embed --bin compare_ort` 走这条路径。

## 做了什么

替换 `burn-flex` 里「rhs 整表转置 + 逐行 i32 点积」：

- 新文件 `crates/burn-flex/src/ops/int_gemm.rs`：row-major `C += A@B`，i-k-j 分块（KC=64, MC=64, NC=128）。
- `u8/i8` 在内积里才 widen 成 i32，不再 `int_cast` 整张激活/权重。
- `M*N*K ≥ 8e6` 且 `M > 64` 时按 MC 条带 rayon 切 M（e5 512 tok 的 Q/FFN 会切；16 tok 通常不切）。
- 累加用 wrapping i32，与旧 kernel / ONNX MatMulInteger 一致。
- 无新增 `unsafe`。

## 对拍数字

见 `notes/poc-results.md`「flex 加速」一节（跑完 `compare_ort` 后填写）。
