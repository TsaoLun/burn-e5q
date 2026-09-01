# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "onnxruntime>=1.22.0",
#   "transformers>=4.44.0",
#   "numpy<2",
# ]
# ///
"""Generate ort reference embeddings for the e5-embed PoC.

Runs the *same* int8 ONNX model that inmotion-social serves
(model_qint8_avx512_vnni.onnx) with the HF tokenizer (tokenizer.json),
then mean-pools + L2-normalizes exactly like `rec::text_embed`.

Output: ref_data.json next to this script, consumed by `compare_ort.rs`.
"""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = (
    Path(__file__).resolve().parents[4]
    / "inmotion-social/data/models/multilingual-e5-small"
)
OUT_PATH = Path(__file__).resolve().parents[1] / "ref_data.json"

PASSAGE = "passage: "
QUERY = "query: "

LONG_TEXT = "Night ride along the riverside with friends. 周末滨江夜骑。 " * 55

CASES = [
    (PASSAGE, "周末滨江夜骑 V11，速度很快！"),
    (PASSAGE, "Hello world! Night ride along the river."),
    (PASSAGE, "週末のリバーサイドナイトライド V11"),
    (PASSAGE, "Café naïve résumé 😄 中文 English 日本語"),
    (PASSAGE, "夜骑"),
    (PASSAGE, LONG_TEXT),
    (PASSAGE, "Multiple   spaces\tand\nnewlines here"),
    (QUERY, "骑行头盔怎么选？"),
    (QUERY, "best helmet for night riding"),
    (PASSAGE, ""),  # empty -> zero vector short-circuit
]


def mean_pool_l2(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    masked = last_hidden * mask
    summed = masked.sum(axis=1)
    mask_sum = mask.sum(axis=1)
    mask_sum = np.where(mask_sum == 0.0, 1.0, mask_sum)
    pooled = summed / mask_sum
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.maximum(norm, 1e-12)


def run_session(sess, tokenizer, texts):
    """Tokenize + pad to batch longest (pad_id=1) + forward + mean pool."""
    enc = tokenizer(texts, truncation=True, max_length=512, add_special_tokens=True)
    rows = enc["input_ids"]
    maxlen = max(len(r) for r in rows)
    input_ids = np.full((len(rows), maxlen), 1, dtype=np.int64)
    attention = np.zeros((len(rows), maxlen), dtype=np.int64)
    for i, r in enumerate(rows):
        input_ids[i, : len(r)] = r
        attention[i, : len(r)] = 1
    out = sess.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention,
            "token_type_ids": np.zeros_like(input_ids),
        },
    )
    return mean_pool_l2(out[0], attention), rows, out[0]


def main():
    print(f"Model dir: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    sess = ort.InferenceSession(
        str(MODEL_DIR / "model_qint8_avx512_vnni.onnx"),
        providers=["CPUExecutionProvider"],
    )
    print("Inputs:", [(i.name, i.shape) for i in sess.get_inputs()])

    cases = []
    for prefix, text in CASES:
        if not text.strip():
            cases.append(
                {"prefix": prefix, "text": text, "ids": [], "embedding": [0.0] * 384}
            )
            continue
        emb, rows, last_hidden = run_session(sess, tokenizer, [prefix + text.strip()])
        case = {
            "prefix": prefix,
            "text": text,
            "ids": [int(i) for i in rows[0]],
            "embedding": [float(x) for x in emb[0]],
        }
        # Keep raw last_hidden_state for the first case to diagnose where
        # burn and ort diverge inside the int8 graph.
        if not any("last_hidden" in c for c in cases):
            case["last_hidden"] = [[float(x) for x in row] for row in last_hidden[0]]
        cases.append(case)
        print(f"  {prefix.strip()}: {text[:30]!r} -> {len(rows[0])} tokens")

    # ort latency baseline (same engine family as inmotion-social's ort crate).
    def bench(fn, repeat=5):
        best = float("inf")
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            best = min(best, (time.perf_counter() - t0) * 1e3)
        return best

    passages = [p + t for p, t in CASES if p == PASSAGE and t.strip()]
    latency = {
        "ort_single_ms": bench(lambda: run_session(sess, tokenizer, [passages[0]])),
        "ort_batch8_ms": bench(lambda: run_session(sess, tokenizer, passages), repeat=3),
        "ort_long512_ms": bench(
            lambda: run_session(sess, tokenizer, [PASSAGE + LONG_TEXT.strip()]), repeat=3
        ),
    }
    print(f"ort latency: {latency}")

    OUT_PATH.write_text(
        json.dumps(
            {"model": "model_qint8_avx512_vnni.onnx", "cases": cases, "latency": latency},
            ensure_ascii=False,
        )
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
