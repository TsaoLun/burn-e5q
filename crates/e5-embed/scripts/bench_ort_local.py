#!/usr/bin/env python3
"""Re-measure ort CPU latency on this machine using ref_data.json token ids.

Does not need transformers: ids are already in the reference file.
Runs the same int8 ONNX as e5-embed (model_qint8_avx512_vnni.onnx).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
REF_PATH = ROOT / "ref_data.json"
MODEL_PATH = Path(os.environ.get("E5_MODEL_PATH", ROOT / "models" / "model_qint8_avx512_vnni.onnx"))


def mean_pool_l2(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    masked = last_hidden * mask
    summed = masked.sum(axis=1)
    mask_sum = mask.sum(axis=1)
    mask_sum = np.where(mask_sum == 0.0, 1.0, mask_sum)
    pooled = summed / mask_sum
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.maximum(norm, 1e-12)


def pad_batch(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    maxlen = max(len(r) for r in rows)
    input_ids = np.full((len(rows), maxlen), 1, dtype=np.int64)
    attention = np.zeros((len(rows), maxlen), dtype=np.int64)
    for i, r in enumerate(rows):
        input_ids[i, : len(r)] = r
        attention[i, : len(r)] = 1
    return input_ids, attention


def run_session(sess, rows: list[list[int]]) -> np.ndarray:
    input_ids, attention = pad_batch(rows)
    out = sess.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention,
            "token_type_ids": np.zeros_like(input_ids),
        },
    )
    return mean_pool_l2(out[0], attention)


def bench(fn, repeat=7, warmup=2) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1e3
        times.append(dt)
        best = min(best, dt)
    return best, times


def main():
    data = json.loads(REF_PATH.read_text())
    print("ort version:", ort.__version__)
    print("providers:", ort.get_available_providers())
    print("model:", MODEL_PATH)
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP", "0") or "0")
    so.inter_op_num_threads = int(os.environ.get("ORT_INTER_OP", "0") or "0")
    sess = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    print("inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print(
        "CPUExecutionProvider options:",
        sess.get_provider_options().get("CPUExecutionProvider"),
    )

    passages = [c for c in data["cases"] if c["prefix"] == "passage: " and c["text"].strip()]
    queries = [c for c in data["cases"] if c["prefix"] == "query: " and c["text"].strip()]

    # Numerical check vs stored embeddings
    print("\n=== cosine vs stored ref embeddings ===")
    min_cos = 1.0
    for c in passages + queries:
        emb = run_session(sess, [c["ids"]])[0]
        ref = np.array(c["embedding"], dtype=np.float32)
        cos = float(np.dot(emb, ref))
        min_cos = min(min_cos, cos)
        label = c["text"][:28]
        print(f"  cos={cos:.6f}  {c['prefix'].strip()} {label!r}")
    print(f"  min cos vs stored ref = {min_cos:.6f}")

    short = [passages[0]["ids"]]
    batch = [c["ids"] for c in passages]
    long = [max(passages, key=lambda c: len(c["ids"]))["ids"]]
    print(f"\nshapes: single={len(short[0])} toks, batch={len(batch)} rows lens={[len(r) for r in batch]}, long={len(long[0])}")

    single_best, single_times = bench(lambda: run_session(sess, short))
    batch_best, batch_times = bench(lambda: run_session(sess, batch), repeat=5, warmup=1)
    long_best, long_times = bench(lambda: run_session(sess, long), repeat=5, warmup=1)

    print("\n=== local ort latency (best of N, after warmup) ===")
    print(f"  single short: {single_best:8.2f} ms  samples={['%.1f' % t for t in single_times]}")
    print(f"  batch {len(batch)}:     {batch_best:8.2f} ms  samples={['%.1f' % t for t in batch_times]}")
    print(f"  long 512:     {long_best:8.2f} ms  samples={['%.1f' % t for t in long_times]}")
    print("\nref_data.json Intel-Mac baseline:")
    print(f"  {data['latency']}")


if __name__ == "__main__":
    main()
