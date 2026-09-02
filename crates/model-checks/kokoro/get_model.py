#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime==1.19.2",
#   "numpy",
# ]
# ///

"""
Download and prepare the Kokoro TTS ONNX model for testing.

Kokoro is a small (82M param) text-to-speech model producing 24kHz audio from
phoneme token IDs and a 256-d voice style embedding.

See: https://github.com/thewh1teagle/kokoro-onnx
"""

import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_artifacts_dir

MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)


def extract_node_info(model_path, artifacts_dir):
    """Extract node types and configurations from the ONNX model."""
    print("Extracting node information from ONNX model...")

    model = onnx.load(str(model_path), load_external_data=False)

    node_types = defaultdict(int)
    node_details = []

    def process_graph(graph, graph_name="main"):
        for idx, node in enumerate(graph.node):
            node_types[node.op_type] += 1
            node_info = {
                "graph": graph_name,
                "index": idx,
                "op_type": node.op_type,
                "name": node.name if node.name else f"{node.op_type}_{idx}",
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {},
            }
            for attr in node.attribute:
                attr_name = attr.name
                if attr.HasField("f"):
                    node_info["attributes"][attr_name] = float(attr.f)
                elif attr.HasField("i"):
                    node_info["attributes"][attr_name] = int(attr.i)
                elif attr.HasField("s"):
                    node_info["attributes"][attr_name] = (
                        attr.s.decode("utf-8") if attr.s else ""
                    )
                elif attr.HasField("t"):
                    node_info["attributes"][attr_name] = "<tensor>"
                elif attr.floats:
                    node_info["attributes"][attr_name] = list(attr.floats)
                elif attr.ints:
                    node_info["attributes"][attr_name] = list(attr.ints)
                elif attr.strings:
                    node_info["attributes"][attr_name] = [
                        s.decode("utf-8") for s in attr.strings
                    ]
                elif attr.HasField("g"):
                    subgraph_name = f"{graph_name}.{node.op_type}_{idx}.{attr_name}"
                    node_info["attributes"][attr_name] = f"<subgraph: {subgraph_name}>"
                    process_graph(attr.g, subgraph_name)
                elif attr.graphs:
                    names = []
                    for g_idx, subgraph in enumerate(attr.graphs):
                        sn = f"{graph_name}.{node.op_type}_{idx}.{attr_name}_{g_idx}"
                        names.append(sn)
                        process_graph(subgraph, sn)
                    node_info["attributes"][attr_name] = f"<subgraphs: {', '.join(names)}>"
                else:
                    node_info["attributes"][attr_name] = "<unknown>"
            node_details.append(node_info)

    process_graph(model.graph, "main")

    summary = {
        "model_name": model.graph.name,
        "opset_version": model.opset_import[0].version if model.opset_import else "unknown",
        "total_nodes": len(node_details),
        "node_type_counts": dict(sorted(node_types.items())),
        "nodes": node_details,
    }

    output_path = artifacts_dir / "node_info.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Opset version: {summary['opset_version']}")
    print(f"  Total nodes: {summary['total_nodes']}")
    print(f"  Unique node types: {len(node_types)}")


def generate_reference_outputs(model_path, artifacts_dir):
    """Generate a deterministic reference test case using ONNX Runtime."""
    print("Creating ONNX Runtime session (this can take a minute)...")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    # Kokoro expects phoneme token IDs in [0, 177] (vocab = 178). Token 0 is
    # the pad/boundary token and is placed at the start and end of the sequence.
    # We don't need semantically meaningful phonemes for an I/O smoke test
    # against ORT, only determinism. Use a fixed seed.
    rng = np.random.default_rng(42)
    inner = rng.integers(low=1, high=178, size=8, dtype=np.int64)
    tokens = np.concatenate([[0], inner, [0]]).reshape(1, -1).astype(np.int64)

    # Style vector is a 256-d voice embedding. Real voice packs produce values
    # roughly in [-0.5, 0.5]. A small scale stays in-distribution.
    style = (rng.standard_normal((1, 256)) * 0.1).astype(np.float32)
    speed = np.array([1.0], dtype=np.float32)

    print(f"  tokens shape: {tokens.shape}")
    print(f"  style shape: {style.shape}")
    print("Running reference inference...")
    outputs = session.run(None, {"tokens": tokens, "style": style, "speed": speed})
    audio = outputs[0].astype(np.float32)

    print(
        f"  audio: shape={audio.shape}, min={audio.min():.4f}, "
        f"max={audio.max():.4f}, mean={audio.mean():.4f}, std={audio.std():.4f}"
    )

    reference = {
        "tokens": tokens.flatten().tolist(),
        "style": style.flatten().tolist(),
        "speed": float(speed[0]),
        "audio_length": int(audio.shape[0]),
        "audio": audio.tolist(),
        "audio_stats": {
            "min": float(audio.min()),
            "max": float(audio.max()),
            "mean": float(audio.mean()),
            "std": float(audio.std()),
        },
    }

    output_path = artifacts_dir / "reference_outputs.json"
    with open(output_path, "w") as f:
        json.dump(reference, f)

    size_kb = output_path.stat().st_size / 1024
    print(f"  Reference saved to {output_path} ({size_kb:.1f} KB)")


def download_model():
    """Download the Kokoro v1.0 ONNX model."""
    artifacts_dir = get_artifacts_dir("kokoro")
    model_path = artifacts_dir / "kokoro-v1.0.onnx"

    if model_path.exists():
        print(f"Model already exists at {model_path}")
        print(f"  File size: {model_path.stat().st_size / (1024 * 1024):.1f} MB")
    else:
        print(f"Downloading Kokoro v1.0 from:\n  {MODEL_URL}")
        print(f"Saving to: {model_path}")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"  File size: {model_path.stat().st_size / (1024 * 1024):.1f} MB")

    extract_node_info(model_path, artifacts_dir)
    print()
    generate_reference_outputs(model_path, artifacts_dir)

    print()
    print("=" * 70)
    print("Model preparation complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  cargo build")
    print("  cargo run")


if __name__ == "__main__":
    download_model()
