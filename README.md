# burn-e5q

Replace ONNX Runtime inference of `intfloat/multilingual-e5-small` (int8) in
inmotion-social with the Burn stack.

This repo is the **orchestration + PoC** workspace. Kernel and codegen work
lands on the forks, then this workspace pins those commits:

| Fork | Branch / pin | Role |
|---|---|---|
| [TsaoLun/burn-onnx](https://github.com/TsaoLun/burn-onnx) | `add-dynamic-quantize-linear` | DQL operator + graph clone-tracking fix |
| [TsaoLun/cubek](https://github.com/TsaoLun/cubek) | `main` (patched over tracel-ai/cubek) | Stage 4: i8 GEMM / VNNI kernels |
| tracel-ai/burn | rev `af844911` | Runtime (flex today; `cpu` for cubecl) |

**Start here for the next agent:** [AGENTS.md](AGENTS.md), then [PLAN.md](PLAN.md) and [notes/stage-4.md](notes/stage-4.md).

## Status

- Stages 0–3 done. Int8 graph **imports and runs**. Tokenizer matches HF.
- Int8 cosine vs ort ≈ **0.996** (expected cross-engine int8 divergence; FP32 model-check is 1e-4 exact).
- Flex backend is **19–45× slower** than ort VNNI. That is stage 4.
- 4096-token-budget peak RSS 640 MB (over 512 MB); 2048-token budget fits at 416 MB.

Details: [notes/poc-results.md](notes/poc-results.md).

## Quick commands

```bash
# DQL smoke (needs crates/dql-poc/models/dql_matmul.onnx)
cargo run --release -p dql-poc

# E5 embed vs ort (needs the int8 ONNX + tokenizer; see AGENTS.md)
cargo run --release -p e5-embed --bin compare_ort
```
