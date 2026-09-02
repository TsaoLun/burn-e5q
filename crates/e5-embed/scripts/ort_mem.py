# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "onnxruntime>=1.22.0",
#   "numpy<2",
# ]
# ///
"""Measure the *entire* ONNX Runtime process RSS on the same shapes as burn.

Mirrors `mem_stress -- 5 2048` (4×512) plus the compare_ort short / 512
forwards. Dummy int64 inputs — no HF tokenizer — so the numbers are the
host + libonnxruntime + weights + scratch, not CPython transformers.

This is the process-level counterpart to burn's `mem_stress` / `compare_ort`
RSS lines (what a 512 MB cgroup actually charges). Subtract the
"numpy + onnxruntime" line to isolate session+weights+scratch.

Usage:
  python3 crates/e5-embed/scripts/ort_mem.py
  python3 crates/e5-embed/scripts/ort_mem.py --arena   # default ORT CPU arena
  python3 crates/e5-embed/scripts/ort_mem.py --rounds 5 --budget 2048
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1] / "models" / "model_qint8_avx512_vnni.onnx"


def rss_mb() -> tuple[float, float]:
    """Return (VmRSS, VmHWM) in MiB from /proc/self/status."""
    rss = hwm = 0.0
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) / 1024.0
    return rss, hwm


def smaps_rss_mb(*needles: str) -> dict[str, float]:
    """Sum mapping Rss (MiB) whose pathname contains a needle."""
    out = {n: 0.0 for n in needles}
    path = ""
    try:
        f = open("/proc/self/smaps", encoding="utf-8", errors="replace")
    except OSError:
        return out
    with f:
        for line in f:
            if line[:1] in "0123456789abcdef" and "-" in line[:20]:
                parts = line.split()
                path = parts[-1] if len(parts) >= 6 else ""
                continue
            if line.startswith("Rss:"):
                kb = int(line.split()[1])
                for n in needles:
                    if n in path:
                        out[n] += kb / 1024.0
                        break
    return out


def log(label: str) -> tuple[float, float]:
    rss, hwm = rss_mb()
    print(f"{label:28s}  RSS {rss:7.1f} MB   HWM {hwm:7.1f} MB")
    return rss, hwm


def log_maps(label: str) -> None:
    maps = smaps_rss_mb("libonnxruntime", "numpy", "libpython")
    parts = "  ".join(f"{k} {v:.1f}" for k, v in maps.items() if v > 0.05)
    if parts:
        print(f"{label:28s}  maps  {parts} MB")


def make_batch(rows: int, seq: int):
    import numpy as np

    ids = np.ones((rows, seq), dtype=np.int64)
    ids[:, 0] = 0
    ids[:, -1] = 2
    attn = np.ones((rows, seq), dtype=np.int64)
    ttype = np.zeros_like(ids)
    return {
        "input_ids": ids,
        "attention_mask": attn,
        "token_type_ids": ttype,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--budget", type=int, default=2048)
    p.add_argument(
        "--arena",
        action="store_true",
        help="leave ORT CPU memory arena on (default: off, like inmotion-social)",
    )
    args = p.parse_args()

    snapshots: list[tuple[str, float, float]] = []

    def snap(label: str) -> tuple[float, float]:
        rss, hwm = log(label)
        snapshots.append((label, rss, hwm))
        return rss, hwm

    snap("python start")
    import numpy as np
    import onnxruntime as ort

    snap("after numpy + onnxruntime")
    log_maps("  file-backed maps")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = args.arena
    so.enable_mem_pattern = args.arena
    snap("after SessionOptions")

    t0 = time.perf_counter()
    sess = ort.InferenceSession(
        str(MODEL),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    load_ms = (time.perf_counter() - t0) * 1e3
    onnx_mb = MODEL.stat().st_size / (1024.0 * 1024.0)
    print(
        f"session loaded in {load_ms:.0f} ms  "
        f"arena={'on' if args.arena else 'off'}  "
        f"ort {ort.__version__}  {ort.get_available_providers()}  "
        f"onnx file {onnx_mb:.1f} MB"
    )
    snap("after InferenceSession")
    log_maps("  file-backed maps")

    short = make_batch(1, 16)
    long = make_batch(1, 512)
    rows = max(args.budget // 512, 1)
    stress = make_batch(rows, 512)
    print(f"stress batch: {rows} rows x 512 tokens (budget {args.budget})")

    t0 = time.perf_counter()
    sess.run(None, short)
    print(f"single 16 tok: {(time.perf_counter() - t0) * 1e3:8.1f} ms")
    snap("after 16-tok forward")

    t0 = time.perf_counter()
    sess.run(None, long)
    print(f"single 512 tok: {(time.perf_counter() - t0) * 1e3:8.1f} ms")
    snap("after 512-tok forward")

    peak_rss = 0.0
    peak_hwm = 0.0
    for i in range(args.rounds):
        t0 = time.perf_counter()
        sess.run(None, stress)
        elapsed = (time.perf_counter() - t0) * 1e3
        rss, hwm = rss_mb()
        peak_rss = max(peak_rss, rss)
        peak_hwm = max(peak_hwm, hwm)
        print(f"round {i:2d}: {elapsed:8.1f} ms, RSS {rss:7.1f} MB  HWM {hwm:7.1f} MB")
    snap("end")

    print()
    print(f"peak observed RSS: {peak_rss:.1f} MB (container budget: 512 MB)")
    print(f"kernel peak HWM:    {peak_hwm:.1f} MB")
    print(
        "verdict: {}".format(
            "within budget" if peak_rss <= 512.0 else "EXCEEDS budget — see notes/poc-results.md"
        )
    )
    print()
    print("snapshot table (entire Python + ORT process):")
    print(f"{'stage':28s}  {'RSS':>8s}  {'HWM':>8s}")
    for label, rss, hwm in snapshots:
        print(f"{label:28s}  {rss:8.1f}  {hwm:8.1f}")

    # Keep numpy referenced so the import is not optimized away by linters.
    _ = np.__version__


if __name__ == "__main__":
    main()
