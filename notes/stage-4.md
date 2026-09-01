# 阶段 4：cubek i8 GEMM 性能攻坚

云端 agent 的操作手册。背景与数字见 `notes/poc-results.md`，仓库边界见 `AGENTS.md`。

## 为什么现在做

flex 后端把 `MatMulInteger` 变成 **I32 朴素三重循环**（`burn-flex/src/ops/matmul.rs`）。
ort 在 x86_64 上走 AVX512-VNNI（`vpdpbusd`：一条指令 4 路 u8×i8→i32 点积）。
实测（Intel Mac，release）：

| 场景 | burn flex | ort | 倍数 |
|---|---|---|---|
| 单条短文本 | 192 ms | 4.3 ms | ~45× |
| 单条 512 tok | 4.5 s | 201 ms | ~22× |

e5 int8 图有 **96 个 MatMulInteger**（与 96 个 DQL 成对）。这是延迟的全部故事。

## 不要走的弯路

1. **不要**在 flex 里手写 AVX kernel。目标后端是 `burn/cpu`（CubeCL JIT → cubek）。
2. **不要**把 DQL 输出先 dequant 成 f32 再 GEMM。要保住 int8 算力。
3. **不要**把 cubek-quant 的 Q4S/Q8S *per-block scale* 方案直接套到 ONNX
   `MatMulInteger`。ONNX 语义是 `(A - a_zp) @ (B - b_zp)`，累加 **I32**，scale
   在 matmul **之后**（DQL 的 `y_scale` 用于反量化）。cubek-quant 的
   `mm_scaled` 是「值 × 块 scale 再乘」，是另一条路。
4. **可以**复用 `cubek-matmul` 的 tiled / `cpu_gemm` 骨架（`EL`/`ER`/`EA` 已经是
   混精度：`crates/cubek-matmul/src/tiled/cpu_gemm/kernel.rs`）。目标是让
   `EL=u8|i8, ER=u8|i8, EA=i32` 的收缩走 VNNI 指令，而不是先 cast 到 i32。

## 推荐拆分（每个都可独立推到 fork）

### 4.1 cubek：整数 GEMM kernel（主战场）

仓库：[TsaoLun/cubek](https://github.com/TsaoLun/cubek)，从 `main` 拉分支
`add-i8-gemm`（名称可改，不要直接推 `main`）。

建议落点（选一，先读再写）：

- 扩展 `crates/cubek-matmul/src/tiled/cpu_gemm/`，让 `cpu_gemm_kernel` 在
  `EA=i32` 且输入是 8-bit 时走 VNNI 友好的 `mm`
- 或在 `tiled/` 下新增 `integer_gemm/`，Blueprint/Routine 对齐
  `cubek/GUIDE.md`

验收（cubek 自己的测试，不依赖 burn-e5q）：

- 正确性：小矩阵 `u8×i8→i32`、`u8×u8→i32`、带/不带 zero-point，对照朴素循环
- 形状：M,N,K 非对齐（e5 里常见 `K=384` / `K=1536`，`M=batch*seq`）
- 若本机有 AVX512-VNNI：同一尺寸相对朴素 i32 GEMM 应有数量级加速
- 无 VNNI 的机器：正确性必须过，可走现存标量/SIMD 回退

CubeCL 现状：`cubecl-core` 已有整数 `Dot`/`SDotOp`，**没有** `vpdpbusd` 助记符。
若 JIT 出不了 VNNI：

1. 先把 i8→i32 的分块 GEMM 做对（即使只是 SIMD 点积），e5 仍会比 flex 快
2. 需要新指令时再 fork [cubecl](https://github.com/tracel-ai/cubecl)（本 workspace
   **尚未** patch cubecl）。不要在 burn-e5q 里塞 `unsafe` 内联汇编。

推送后把 `burn-e5q/Cargo.toml` 里两处 cubek `rev` 一起改掉，跑
`cargo update -p cubek`。

### 4.2 burn-cpu 接线

文件：`tracel-ai/burn` @ `af844911`

```
crates/burn-cubecl/src/ops/int_tensor.rs   # CubeBackend::int_matmul
crates/burn-cubecl/src/kernel/matmul.rs    # MatmulStrategy
```

今天：

```rust
fn int_matmul(lhs, rhs) {
    matmul(lhs, rhs, None, MatmulStrategy::default(), lhs.dtype).unwrap()
}
```

要做的：当 `lhs/rhs` 是 `I8`/`U8`、累加 I32 时，调 4.1 的 kernel；其余 dtype
保持原路径。

本 workspace **没有 burn fork**。可选：

- 开 `TsaoLun/burn` 并在 `Cargo.toml` 把 `burn` / `burn-store` 的 git 改过去
- 或 `[patch."https://github.com/tracel-ai/burn"]`（必须列出用到的所有 burn 包，
  很烦；优先整个 git rev 切换）

没有接线时，cubek 测试仍能证明 kernel；`e5-embed --features cpu` 不会变快。

### 4.3 burn-onnx codegen（可能不必改）

`crates/burn-onnx/src/import/burn/node/matmul_integer.rs` 现在把输入 **cast 成
I32 再 `.matmul()`**。若 4.2 只看运行时 dtype，这条路径会继续走慢的 i32 GEMM。

两种修法（选更干净的一种）：

- **A（推荐）**：codegen 保留 u8/i8 dtype，zp 用 i32 减法或 fused kernel 参数，
  让 `.matmul()` 落在 i8 kernel 上
- **B**：backend 提供显式 `matmul_integer(lhs, rhs, zp_a, zp_b)`，codegen 调它

改动推到 [TsaoLun/burn-onnx](https://github.com/TsaoLun/burn-onnx) 的
`add-dynamic-quantize-linear`（DQL 已在此分支）或新分支，然后 bump
`burn-e5q` 的 `burn-onnx` / `onnx-ir` `rev`。

### 4.4 用 e5-embed 回归

```bash
# 需要 int8 ONNX + tokenizer，见 AGENTS.md
cargo run --release -p e5-embed --features cpu --no-default-features --bin compare_ort
cargo run --release -p e5-embed --features cpu --no-default-features --bin mem_stress -- 5 4096
```

记录到 `notes/poc-results.md`（新开一节「阶段 4」，不要删阶段 3 数字）：

- 延迟 vs 表中 ort 基线
- min/mean cosine（预期仍约 0.996，不要为了追 0.999 改 DQL round）
- peak RSS vs 512 MB

## e5 图里 MatMulInteger 的形状（调 kernel 时对着它）

XLM-RoBERTa-small，int8 ONNX，96 个整数 matmul，大致两类：

- Attention：`[B*heads, seq, head_dim=64]` × `[B*heads, 64, seq]`
- FFN：`[B, seq, 384] × [384, 1536]` 与 `[B, seq, 1536] × [1536, 384]`

DQL 输出 **U8** + 标量 scale/zp。Zero-point 在 MatMulInteger 的可选输入上。

## 本地 cubek 迭代（可选）

```bash
git clone https://github.com/TsaoLun/cubek.git
# 在 burn-e5q/Cargo.toml：注释 git [patch]，打开 path [patch]
# 改 kernel → 在 cubek 仓库跑其测试 → commit push → 改回 git rev
```
