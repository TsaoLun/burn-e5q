# AGENTS.md — burn-e5q

PoC workspace for replacing inmotion-social's ort E5-small int8 inference with
Burn. **Stages 0–3 are done.** The next job is **stage 4: i8 GEMM performance**.

Read this file first, then `PLAN.md` and `notes/stage-4.md`. Do not re-implement
DynamicQuantizeLinear or the e5-embed pipeline unless a regression appears.

## What this repo is (and is not)

- **This repo** (`burn-e5q`): tokenizer + mean-pool + L2 + ort comparison +
  memory stress. Integration benchmark after kernel work.
- **Not this repo:** ONNX codegen lives in [TsaoLun/burn-onnx](https://github.com/TsaoLun/burn-onnx);
  kernels live in [TsaoLun/cubek](https://github.com/TsaoLun/cubek).
- There is **no path dependency** on sibling checkouts. Cargo pulls the forks
  over git. After you push a fork, bump the `rev` in the root `Cargo.toml`
  (`workspace.dependencies` **and** `[patch."https://github.com/tracel-ai/cubek"]`).

## Fork pins (keep in sync)

| Crate | Source | Current pin |
|---|---|---|
| `burn-onnx`, `onnx-ir` | `https://github.com/TsaoLun/burn-onnx` | `63e35840812fe573608d7152868f2c1972494887` (`add-dynamic-quantize-linear`) |
| `cubek` | `https://github.com/TsaoLun/cubek` via `[patch]` of `tracel-ai/cubek` | `c1a1a9eb5e655d8728d92e61b8a44ce0794d9afb` (`main`) |
| `burn`, `burn-store` | `https://github.com/tracel-ai/burn` | `af844911be6efb6745301c1c2c5e695d6571b316` |

`burn/cpu` still depends on `tracel-ai/cubek`. The `[patch]` table is what makes
`cargo run -p e5-embed --features cpu` pick up the fork. Do not delete it.

Optional local overlay while iterating: clone cubek as a sibling, comment out
the git `[patch]` block, uncomment the `path = "../cubek/crates/cubek"` block.
Cargo allows only one `[patch]` table per source URL.

## Stage 4 goal

Cut e5-embed latency from ~20–45× slower than ort down toward parity, by
giving CubeCL-CPU a real `u8/i8 → i32` GEMM (AVX512-VNNI `vpdpbusd` on x86_64)
instead of flex's naive i32 triple loop.

Work order (details in `notes/stage-4.md`):

1. **cubek** — integer GEMM kernel + tests on TsaoLun/cubek. Push a branch, bump
   this workspace's cubek `rev`.
2. **burn-cpu wiring** — `CubeBackend::int_matmul` currently calls generic
   `matmul(...)`. Route I8/U8→I32 through the new kernel. This lives in
   `tracel-ai/burn` (`crates/burn-cubecl/src/ops/int_tensor.rs`). If you need
   to persist that change, fork burn or `[patch]` it; this workspace does not
   yet fork burn.
3. **burn-onnx (optional later)** — `MatMulInteger` codegen today casts to I32
   then `.matmul()`. Once the backend's int matmul is fast, codegen may not
   need to change. Only special-case codegen if zp/u8×i8 layout requires it.
4. **Re-bench** — `cargo run --release -p e5-embed --features cpu --bin compare_ort`
   and `mem_stress`. Target: within ~2× of the ort baseline in `ref_data.json`
   (single short ~4 ms, 512-token ~200 ms on the machine that wrote that file).

## How to run the PoC

Artifacts are **not** in git (~118 MB ONNX). Resolve them in this order:

1. `E5_MODEL_PATH` / `E5_MODEL_DIR`
2. `crates/e5-embed/models/` (`model_qint8_avx512_vnni.onnx` +
   `sentencepiece.bpe.model` + `config.json` + `tokenizer_config.json`)
3. Sibling `../inmotion-social/data/models/multilingual-e5-small/` (local only)

HF source: `intfloat/multilingual-e5-small`, file
`onnx/model_qint8_avx512_vnni.onnx`. Tokenizer: `sentencepiece.bpe.model`.

```bash
# flex (correctness; slow)
cargo run --release -p e5-embed --bin compare_ort

# cubecl CPU (the stage 4 target)
cargo run --release -p e5-embed --features cpu --no-default-features --bin compare_ort

# memory: ./mem_stress <rounds> <token_budget>
cargo run --release -p e5-embed --bin mem_stress -- 5 2048
```

`crates/e5-embed/scripts/gen_ref.py` regenerates `ref_data.json` (needs
onnxruntime + transformers; HTTP proxy if HF is blocked).

## Constraints

- Respond in 中文 if the user writes 中文.
- No `unsafe` in burn-onnx. cubek/cubecl already use CubeCL codegen rather than
  hand-written `unsafe` SIMD; prefer that. Do not add `unsafe` to burn-e5q.
- Do not commit the 118 MB ONNX or `.bpk` files.
- Do not open PRs against tracel-ai unless the user asks. Push to the TsaoLun
  forks. Do not `git push --force` to `main`.
- Official DQL node tests compile but have **no numerical harness** (rank-0
  scalar outputs). Correctness: `dql-poc`, insta snapshots in burn-onnx, and
  this crate's ort compare.
- macOS x86_64: `ort` crate has no prebuilt `ort-sys` → `cargo test -p burn-onnx`
  lib tests may fail to link. `onnx-official-tests` does not need ort.

## Do not redo

- DQL `NodeProcessor` / `NodeCodegen` (fork branch `add-dynamic-quantize-linear`).
- `graph.rs` boundary scalar future-use registration (`63e35840`).
- e5-embed tokenizer (HF id remap, prefixes, pack_batches, mean-pool + L2).
- multilingual-e5-small **FP32** model-check (separate branch
  `add-multilingual-e5-small-model-check`; not a runtime dep of this workspace).
