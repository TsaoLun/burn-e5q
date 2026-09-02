# 阶段 4 实现：i8 GEMM 栈

> 2026-09-01。TsaoLun/{cubek,burn,burn-onnx,cubecl} 对本 agent 的 `git push` 返回 **403**，
> 所以每个 fork 的工作树以 orphan snapshot 发在本仓库的 `vendor/*` 分支上，供 Cargo `rev` 钉住。
> 你本地有写权限后，把同样的提交推到对应 fork，再把根 `Cargo.toml` 的 `rev` 改回去。

## 做了什么

### 1. cubek `vendor/cubek-add-i8-gemm` (`29485715`)

- CpuGemm 叶子本来就会把 `EL`/`ER` cast 进 `EA`；`u8/i8 → i32` 走同一条分块路径。
- `select()`：L1 的 K 面板按 **输入** 字节计（8-bit 能更深），累加器预算仍按 i32；
  两边都是 1 字节时优先 `tile_k` 为 4 的倍数（VNNI 友好）。
- 测试（`cargo test -p cubek-matmul --release --features cubecl/cpu --test lib cpu_int8`）：
  `u8×u8`、`i8×i8`、`u8×i8`、K 非 4 对齐、e5-like 8×32×64、Inferred heuristic。**6/6 过。**

### 2. burn `vendor/burn-route-int8-matmul` (`2223f5a0`)

- `CubeBackend::int_matmul`：两边都是 I8/U8 时输出 **I32**，策略 **CpuGemm**（不走 autotune）。
- `launch_matmul` 修了 rhs 被标成 `lhs.dtype` 的 bug（否则 u8×i8 错）。
- flex：`u8/i8`（含混合）先 `int_cast(I32)` 再走现有 i32 GEMM。单测 `u8×i8` / `u8×u8` 过。

### 3. burn-onnx `vendor/burn-onnx-keep-int8-matmul` (`39a7d6ff`)

- MatMulInteger **不再** `cast(I32)` 再 `.matmul()`。
- zp 用代数恒等式，乘加仍在 I32：
  `(A-za)@(B-zb) = A@B − za·sum_k(B) − sum_k(A)·zb + za·zb·K`
- insta 10/10 过（含新的 u8×i8 锁测试）。

### 4. cubecl `vendor/cubecl-host-native-jit` (`a62bcd86`)

- `cubecl-llvm` CPU JIT：`LLVMRunPasses(..., tm=null)` 改成 host `TargetMachine`
  （`LLVMGetHostCPUName` / `Features` + Aggressive + JITDefault），并套上 TM 的 data layout。
- 没有 TM 时回退 null，避免非本机 triple 编不过。
- `cargo check -p cubecl-llvm` 过。这是让 O3 有机会打出 AVX512/VNNI 的前提；
  叶子仍是 tiled `SUM_PROD`，不是手写 `vpdpbusd`。

## 预期

三步做完 + host TM：**不会**打平本机 AMX ort。长序列有机会靠近 `ref_data.json` 的 Mac ort（~2×）；
短序列仍可能被 ~1630 次 launch + 96 次 DQL 卡住。

## 把 snapshot 迁回真 fork

```text
cubek     add-i8-gemm              本地 /tmp/forks/cubek
burn      route-int8-matmul        本地 /tmp/forks/burn
burn-onnx keep-int8-matmul         本地 /tmp/forks/burn-onnx
cubecl    host-native-jit          本地 /tmp/forks/cubecl
```

推上去之后改根 `Cargo.toml` 的 git URL + `rev`，并保留 `[patch."https://github.com/tracel-ai/cubek"]`。
