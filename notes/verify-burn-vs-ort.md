# 复测：burn-onnx / flex 相对 ort 的延迟差距

> 日期：2026-09-01。环境：Linux x86_64 云主机，4 核 Intel Xeon（KVM），
> 标志含 `avx512_vnni`、`avx_vnni`、`amx_int8`。rustc 1.98（workspace edition 2024，1.83 编不了）。
> burn：flex 后端，release，`e5-embed` `compare_ort`。
> 对照：`ref_data.json` 里的 Intel Mac ort 基线，以及本机 Python onnxruntime 1.29。

## 结论（先看这个）

文档里「flex 比 ort 慢 **19–45×**」在本机 **成立，而且在有 VNNI/AMX 的 Linux 上更大**。

| 场景 | burn flex | Mac ort（`ref_data.json`） | 相对 Mac | 本机 ort 1.29 | 相对本机 |
|---|---:|---:|---:|---:|---:|
| 单条短文本（16 tok） | 129.7 ms | 4.3 ms | **30×** | 3.43 ms | **38×** |
| batch 7–8（含 512） | 26.2 s | 1.41 s | **19×** | 530 ms | **49×** |
| 单条 512 tok | 3.82 s | 201 ms | **19×** | 49.9 ms | **77×** |

Mac 上的 19–45× 是同一套图在「无/弱 VNNI 利用的 i32 三重循环」对「VNNI 量化 GEMM」的差距。
本机 ort 还能吃 AMX-INT8，512-token 从 201 ms 掉到 50 ms，**分母变小，倍数被拉大**。

数值侧与阶段 3 一致：mean cos **0.9960**（min 0.9935），top-1 检索仍对，top-3 两位对调。
这不是性能 bug。同模型、同 ids 的本机 ort vs Mac 存档向量已经只有 min cos 0.9948——int8 跨实现/跨平台本来就不 bit-exact。

## 怎么复现

模型不在 git。从 HF `intfloat/multilingual-e5-small` 取
`onnx/model_qint8_avx512_vnni.onnx` + `sentencepiece.bpe.model` + `config.json`，放到
`crates/e5-embed/models/`。

```bash
# 本机 ort 基线（用 ref_data.json 里已有的 token ids，不需要 transformers）
python3 crates/e5-embed/scripts/bench_ort_local.py

# burn flex（阶段 3 / 当前默认）
cargo run --release -p e5-embed --bin compare_ort

# 单核 i32 vs packed VNNI 微基准（e5 FFN/QKV 形状）
gcc -O3 -march=native -o /tmp/gemm_microbench crates/e5-embed/scripts/gemm_microbench.c
/tmp/gemm_microbench
```

`cargo run --release -p e5-embed --features cpu --no-default-features --bin compare_ort`
**当前编不过**（见文末）。阶段 4 的 cubecl 路径还没有可跑的对照数字。

## 图里实际在算什么

生成代码：`target/release/build/e5-embed-*/out/model/model_qint8_avx512_vnni.rs`（12646 行，13 个 submodule）。

| 生成物 | 数量 |
|---|---:|
| `.matmul(` / `let matmulinteger` | **96** |
| `let dynamicquantizelinear`（每个 DQL 三个输出） | 288 = 96×3 |
| `DType::I32` 出现次数 | 390 |

配置：`hidden_size=384`，`num_attention_heads=12`，`intermediate_size=1536`，12 层。
生成代码把 Q/K/V reshape 成 `[B, S, 12, 32]`，所以 **head_dim = 32**（`notes/stage-4.md` 里写的 64 是笔误；384/12=32）。

每层 8 个 MatMulInteger：Q/K/V、attn scores、attn context、out proj、FFN1、FFN2。12×8=96。

整数 GEMM 工作量（只计这些 MatMulInteger 的 MAC）：

| 形状 | 16 tok | 512 tok |
|---|---:|---:|
| QKV 3× `[S,384]×[384,384]` | 0.085 G | 2.72 G |
| out `[S,384]×[384,384]` | 0.028 G | 0.91 G |
| FFN `[S,384]×[384,1536]` + 回投 | 0.226 G | 7.25 G |
| attn 12× `[S,32]×[32,S]` 两次 | 0.002 G | 2.42 G |
| **合计** | **0.34 GMAC** | **13.3 GMAC** |

本机有效吞吐：

- flex 512-tok：13.3 GMAC / 3.82 s ≈ **3.5 GMAC/s**
- ort 512-tok：13.3 GMAC / 0.050 s ≈ **266 GMAC/s**（约 76×）

4 核 2.4 GHz 粗峰值：AVX512 i32 FMA ≈ 154 GMAC/s；`vpdpbusd`（每条 64 个 u8×i8 MAC）≈ 614 GMAC/s。
ort 大约吃到 VNNI 峰值的 40%+，再叠加 AMX 也说得通。flex 连单核 i32 峰值（~38 GMAC/s）都只到约 10%。

## 根因（按贡献排序）

### 1. `MatMulInteger` 被降成 I32 GEMM，丢掉了 int8 算力

burn-onnx codegen（`matmul_integer.rs`）对每个整数 matmul 做的是：

```text
(A.cast(I32) - zp_a) .matmul (W.cast(I32) - zp_b)   → I32 → .float() → × scale
```

生成代码里第一层 Q 投影就是这样（权重每次 forward 都从 U8/I8 cast 成 I32）：

```rust
let matmulinteger1_out1 = ((dynamicquantizelinear1_out1.clone())
    .cast(burn::tensor::DType::I32)
    .sub(dynamicquantizelinear1_out3.clone().cast(DType::I32).unsqueeze::<3>()))
    .matmul(
        constant135.cast(DType::I32)
            .sub(constant134.cast(DType::I32).unsqueeze::<2>())
            .unsqueeze::<3>(),
    );
```

后果：

- 运行时 dtype 是 I32，**即使以后 cubek 有 u8×i8 kernel，这条路径也走不到**（`notes/stage-4.md` 4.3 已写）。
- 权重/激活带宽变成 4 倍；zp 减法物化成整张 I32 张量。
- 权重是 Constant，却在 **每个 forward、每个 GEMM** 重新 cast，而不是预打包成 VNNI 布局。

### 2. flex 的整数 matmul 是「转置 + 点积」，不是分块 GEMM，更不是 VNNI

`burn-flex/src/ops/matmul.rs`：f32/f16 走 `gemm` crate；**I32 不走 gemm**。

```rust
match lhs.dtype() {
    DType::I32 => matmul_i32(lhs, rhs),
    DType::I64 => matmul_i64(lhs, rhs),
    _ => panic!("int_matmul: unsupported dtype {:?}", lhs.dtype()),
}
```

`matmul_2d_i32`：把 rhs `[K,N]→[N,K]`，然后对每个 `(i,j)` 做 `dot_i32`（simd feature 下是 macerator 的 i32 `vmul_add`）。没有 cache blocking，没有 `vpdpbusd`。

更糟的形状细节：QKV/FFN 的权重是 2D，codegen `unsqueeze::<3>()` 之后变成 **batch=1 的 batched i32**。`matmul_batched_i32` 的 rayon 只切 batch 维，batch=1 时 **整块 FFN GEMM 单核**。注意力是 `[B,12,S,32]`，12 个头才能并行。

本机单核微基准（`scripts/gemm_microbench.c`，flex 同款转置+点积 vs 预打包 `vpdpbusd`）：

| 形状 | i32-flex | packed VNNI | VNNI 加速 |
|---|---:|---:|---:|
| FFN1 `[512,384]×[384,1536]` | 28.0 ms / 10.8 GMAC/s | 4.5 ms / 67 GMAC/s | **6.2×** |
| FFN2 `[512,1536]×[1536,384]` | 31.0 ms / 9.8 GMAC/s | 6.2 ms / 49 GMAC/s | **5.0×** |
| QKV `[512,384]×[384,384]` | 4.8 ms / 15.7 GMAC/s | 1.1 ms / 69 GMAC/s | **4.4×** |
| 短 FFN `[16,384]×[384,1536]` | 5.9 ms / 1.6 GMAC/s | 0.15 ms / 64 GMAC/s | **40×** |

这只是 **单核、无分块** 的 VNNI。ort 还有打包一次、多核、AMX、算子融合，所以端到端能到 ~266 GMAC/s，而不是 67。

短序列 40× 已经接近文档里「单条短文本 45×」：小 M 时 i32 路径固定开销（转置、分配）占比更高，VNNI 仍接近满吞吐。

### 3. 96 个 DQL 是额外的 f32 归约，不是免费的

每个 MatMulInteger 前面都有 DynamicQuantizeLinear：`min`/`max` 扫整张激活、round、clamp、cast U8。
注意力还要对 Q、K、softmax 后再 DQL。这些是 f32 全归约 + 逐元素，短序列时相对 GEMM 更显眼（所以短文本倍数 > 长文本，在 Mac 基线上是 45× vs 22×）。

### 4. 图是 1630 节点的指令式 forward，没有 ORT 那种 EP fusion

LayerNorm / GELU / softmax / 一堆 reshape-permute-cast 都是独立 tensor op。ort 的 CPU EP 会把量化线性层收成一条 MLAS 路径。burn-onnx 按节点展开，中间张量活到语句结束。

## 和文档数字的对照

| 项 | `notes/poc-results.md`（Intel Mac flex） | 本次 Linux flex | 判定 |
|---|---|---|---|
| 短文本 | 192 ms / 45× | 130 ms / 30× vs Mac ort，38× vs 本机 ort | 同数量级；本机 CPU 稍快 |
| 512 tok | 4.5 s / 22× | 3.82 s / 19× vs Mac ort，**77× vs 本机 ort** | flex 几乎没吃到 VNNI/AMX；ort 吃到了 |
| batch | 26.9 s / 19× | 26.2 s / 19× vs Mac ort，49× vs 本机 | 长序列主导，和 512 tok 一致 |
| mean cos | 0.9960 | 0.9960 | 复现 |
| tokenizer | 9/9 | 9/9 | 复现 |
| 加载 RSS | 92.6 MB | 87.8 MB | 复现 |
| `--features cpu` | （未作为阶段 3 主路径） | **burn-cubecl 编不过** | 阶段 4 接线前必须先对齐 cubecl rev |

## `--features cpu` 为什么现在编不过

`cubek` patch 钉在 `c1a1a9eb`，它依赖 cubecl `2f9b71fe`；burn `af844911` 自己用 cubecl `a30c2e31`。
`burn-cubecl` 编译报 107 个错，典型信息：

```text
multiple different versions of crate `cubecl_ir` / `cubecl_common`
expected cubek::cubecl::prelude::ElemType, found cubecl::prelude::ElemType
flex32: MatmulPrecision not implemented for burn_std::flex32
  (implemented for cubek::cubecl::flex32)
```

所以阶段 4 不能只写 kernel：要先让 **TsaoLun/cubek 与 tracel-ai/burn 的 cubecl rev 是同一份**，否则 `e5-embed --features cpu` 无法作为回归入口。

即便编过，codegen 仍会先 `cast(I32)`，`CubeBackend::int_matmul` 也只是把 **I32** 丢给通用 `matmul(...)`。没有「保留 u8/i8」+「路由到整数 GEMM」这两步，cubecl CPU 也追不平 ort。

## 阶段 4 仍然该做的（验证后未改判断）

1. cubek：真正的 `u8/i8 → i32` 分块 GEMM（x86 上 `vpdpbusd`；本机还有 AMX-INT8 可后做）。
2. burn-cpu：`int_matmul` 在 I8/U8 时走该 kernel。
3. burn-onnx：**不要**再默认 cast 到 I32；zp 留在 kernel 参数或 fused 减法。权重应预打包，而不是每个 forward cast。
4. 先把 cubek / cubecl / burn 的 git rev 对齐，否则连 cpu feature 都没有。

目标仍是：短文本接近本机 ~3–4 ms、512 tok 接近本机 ~50 ms（或至少 Mac 基线的 ~200 ms 的 2× 以内）。现在 130 ms / 3.8 s 离这个目标还差一个数量级以上，瓶颈就是上面 1–3，不是 tokenizer，也不是 DQL 语义。
