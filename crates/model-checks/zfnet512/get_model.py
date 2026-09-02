#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime>=1.22.0",
#   "numpy",
#   "torch",
# ]
# ///

"""Download ZFNet-512 from the ONNX Model Zoo."""

import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_artifacts_dir, process_model

MODEL_NAME = "zfnet512"

# ONNX Model Zoo: ZFNet-512, opset 12
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/zfnet-512/model/zfnet512-12.onnx"


def download_model(output_path):
    """Download ZFNet-512 ONNX model from the ONNX Model Zoo."""
    print("Downloading ZFNet-512 from ONNX Model Zoo...")
    urllib.request.urlretrieve(MODEL_URL, output_path)
    file_size = output_path.stat().st_size if output_path.exists() else 0
    if file_size < 1000:
        raise RuntimeError(
            f"Download failed or file too small ({file_size} bytes). "
            "The ONNX Model Zoo may use Git LFS; try cloning the repo instead."
        )
    print(f"  Model downloaded to: {output_path}")


def generate_test_data(model_path, output_dir):
    """Generate test input/output data using ONNX Runtime."""
    import onnxruntime as ort
    import torch

    print("\nGenerating test data...")

    np.random.seed(42)
    # ZFNet-512 expects [1, 3, 224, 224]
    test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: test_input})

    output_names = [o.name for o in session.get_outputs()]
    print(f"  Input name: {input_name}")
    print(f"  Model outputs: {output_names}")
    for i, name in enumerate(output_names):
        print(f"    - {name}: shape={outputs[i].shape}")

    # Normalize keys: replace slashes to avoid PytorchStore key mapping issues
    def normalize_key(key):
        return key.replace("/", "_")

    test_data = {normalize_key(input_name): torch.from_numpy(test_input)}
    for i, name in enumerate(output_names):
        test_data[normalize_key(name)] = torch.from_numpy(outputs[i])

    test_data_path = Path(output_dir) / "test_data.pt"
    torch.save(test_data, test_data_path)
    print(f"  Test data saved to: {test_data_path}")


def main():
    print("=" * 60)
    print("ZFNet-512 Model Preparation")
    print("=" * 60)

    artifacts_dir = get_artifacts_dir(MODEL_NAME)

    raw_path = artifacts_dir / "zfnet512_raw.onnx"
    model_path = artifacts_dir / "zfnet512.onnx"
    test_data_path = artifacts_dir / "test_data.pt"

    if model_path.exists() and test_data_path.exists():
        print(f"\n  All files already exist:")
        print(f"  Model: {model_path}")
        print(f"  Test data: {test_data_path}")
        print("\nNothing to do!")
        return

    if not raw_path.exists() and not model_path.exists():
        print("\nStep 1: Downloading model...")
        download_model(raw_path)

    if not model_path.exists():
        print("\nStep 2: Processing model (upgrading opset, shape inference)...")
        process_model(raw_path, model_path, target_opset=16)
        if raw_path.exists():
            raw_path.unlink()

    if not test_data_path.exists():
        print("\nStep 3: Generating test data...")
        generate_test_data(model_path, artifacts_dir)

    print("\n" + "=" * 60)
    print(f"  ZFNet-512 model preparation completed!")
    print(f"  Model: {model_path}")
    print(f"  Test data: {test_data_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
