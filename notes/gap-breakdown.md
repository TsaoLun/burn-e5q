# 拆 1.12 s：融合 attn 之后还剩什么

> 2026-09-03。同一台 4 核 Xeon（`avx512_vnni` + `amx_int8`）。
> 栈：融合 int8 SDPA → `attention()`（#6）+ flash 按 head 并行（#7）。
> 命令：`cargo run --release -p e5-embed --bin breakdown`
> 基线：本机 Rust ort arena off，512 tok **53.8 ms**。flex 512 **~1.12 s**（21×）。

这一刀**不改模型、不写 kernel**。用同后端、同形状的隔离计时，把 1.12 s 对到具体算子上，再决定下一刀。

## 图里还剩什么（生成代码计数）

`model_qint8_avx512_vnni.rs`（融合 attn + flash 门槛之后）：

| 算子 | 次数 | 512 形状 |
|---|---:|---|
| `module::attention` | 12 | QKV `[1,12,512,32]`，bias `[1,1,1,512]` |
| `matmul_integer` | 72 | 每层 3×QKV + out `[512,384]×[384,384]`；FFN1 `[512,384]×[384,1536]`；FFN2 `[512,1536]×[384]` |
| `dynamic_quantize_linear` | 48 | 36× `[1,512,384]` + 12× `[1,512,1536]` |
| 展开 LayerNorm（`mean_dim` 50 / `powf` 25） | ~25 | `[1,512,384]`（每层 attn 前 + FFN 前，加 embedding/最终 LN） |
| 展开 GELU（`erf` 12） | 12 | `[1,512,1536]` |
| `clone` / `unsqueeze` | 884 / 701 | 胶水 |

没有 Softmax 节点。`[H,S,S]` 不再物化。#7 已经证明剩下的时间**不在 flash**。

## 方法

`crates/e5-embed/src/bin/breakdown.rs`：

1. **隔离块**：flash ×12、MMI QKV+out ×36、FFN1 ×12、FFN2 ×12、DQL ×48、展开 LN ×25、展开 GELU ×12、MMI 反量化 ×72、QKV reshape/permute ×12、embedding `take`+dequant。权重是 dummy u8/i8，dtype 显式标成 U8×I8，走和线上一样的 VNNI 路径。
2. **链式骨架**：按真实一层的顺序串起来（LN → DQL → QKV MMI → attn → out → residual → LN → FFN1 → GELU → FFN2），跑 12 层。
3. **校准**：同一进程里 `E5Embedder::embed_passages` 一条 512 tok、一条短句。
4. flex 是 eager。计时包住算子调用，`black_box` 防 DCE。每块 warmup 1 次，报 min / median。

读法：

| 信号 | 意思 |
|---|---|
| MMI（72 个）占大头 | 下一刀是 FFN/QKV 调度或整层融合，不是 flash |
| LN + GELU + 胶水很大 | 调度税；codegen 收成 `layer_norm` / `gelu`，或整层执行单元 |
| 隔离之和 ≪ 真实 512 | 间隙是生成图税（`.val()`、Shape/Gather/Concat、884 clone） |
| 隔离之和 ≈ 真实，且 MMI 已接近 ORT 该有的算力 | ORT 赢在融合，不是单 kernel 慢 |

**不要**再调 flash TILE / gemm 并行。**不要**先写 int8 flash 或整层融合——等这组数说话。

## 测数

（跑 `breakdown` 之后填。）
