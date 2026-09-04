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
| `burn-onnx`, `onnx-ir` | this repo `vendor/burn-onnx-coalesce-gelu-ln` | `68153cc97a63b4b3c91dbee64f3a747268ada525` |
| `cubek` | this repo `vendor/cubek-add-i8-gemm` via `[patch]` of `tracel-ai/cubek` | `29485715f433fd26863dcaa5c8cc80f2a98f6183` |
| `burn`, `burn-store` | this repo `vendor/burn-simd-gelu` | `a62f53407164d9dfd390fda2bdf8c82c9469bcdc` |
| `cubecl` (transitive) | this repo `vendor/cubecl-host-native-jit` | `a62bcd86aba5b9e530be6abd4d47810d3177d8d0` |

The TsaoLun forks denied this agent's `git push` (403). Each working tree is
an orphan snapshot on **this** repo so Cargo can still pin them. After you
push `add-i8-gemm` / `route-int8-matmul` / `keep-int8-matmul` / `host-native-jit`
to the real forks, retarget the `rev`s in the root `Cargo.toml`.

`burn/cpu` still depends on `tracel-ai/cubek`. The `[patch]` table is what makes
`cargo run -p e5-embed --features cpu` pick up the integer GEMM. Do not delete it.

Optional local overlay while iterating: clone cubek as a sibling, comment out
the git `[patch]` block, uncomment the `path = "../cubek/crates/cubek"` block.
Cargo allows only one `[patch]` table per source URL.

## Stage 4 goal

Cut e5-embed latency from ~20–45× slower than ort down toward parity, by
giving CubeCL-CPU a real `u8/i8 → i32` GEMM (AVX512-VNNI `vpdpbusd` on x86_64)
instead of flex's naive i32 triple loop.

Work order (details in `notes/stage-4.md` and `notes/stage-4-impl.md`):

1. **cubek** — `u8/i8→i32` CpuGemm + tests (`vendor/cubek-add-i8-gemm`).
2. **burn-cpu wiring** — I8/U8 `int_matmul` → I32 via `MatmulStrategy::CpuGemm`;
   flex widens mixed u8×i8 (`vendor/burn-route-int8-matmul`).
3. **burn-onnx** — MatMulInteger keeps input dtypes; zp via
   `(A-za)@(B-zb) = A@B − za·sum_k(B) − sum_k(A)·zb + za·zb·K`
   (`vendor/burn-onnx-keep-int8-matmul`).
4. **cubecl** — host `TargetMachine` for LLVM `default<O3>` so the leaf can
   autovec to AVX512/VNNI (`vendor/cubecl-host-native-jit`).
5. **fused DQL** — `Tensor::dynamic_quantize_linear`; flex two-pass minmax+quantize
   (`vendor/burn-route-int8-matmul` `2d1084f` + codegen `7cc2d36`).
6. **fused int8 attention** — coalesce e5's DQL+MatMulInteger SDPA onto Burn
   `attention()`; flex sends 512×512 through flash and parallelizes heads
   (`vendor/burn-onnx-coalesce-int8-attn` `f78e156` +
   `vendor/burn-flash-par-heads` `fd4f793`).
7. **fused GELU / LayerNorm** — coalesce the expanded erf-GELU and last-axis
   LN subgraphs; flex parallelizes large f32 `gelu` (`vendor/burn-onnx-coalesce-gelu-ln`
   `68153cc` + `vendor/burn-flex-par-gelu` `319336c`).
8. **AMX int8 GEMM** — aligned u8×i8 MMI uses `tdpbusd` via stable `asm!`.
   VNNI-QK flash was measured slower than tiled f32 and is not hooked
   (`vendor/burn-int8-flash-amx` `21dba0c`).
9. **D=32 AVX-512 flash** — long f32 heads with `D=32` use a tiled AVX-512
   QK / softmax / PV kernel; `[1,1,1,S]` masks no longer expand to `[H,S,S]`
   (`vendor/burn-flash-d32` `219fe61`).
10. **AVX-512 GELU** — contiguous f32 GELU uses musl/fdlibm `erff`
    polynomials (not A&S); large buffers still rayon
    (`vendor/burn-simd-gelu` `a62f534`).
11. **Re-bench** — `cargo run --release -p e5-embed --bin compare_ort`
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

# Rust ort (inmotion-social): cosine + latency + process RSS
cargo run --release -p ort-mem -- -- 5 2048
cargo run --release -p ort-mem -- --arena -- 5 2048
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
