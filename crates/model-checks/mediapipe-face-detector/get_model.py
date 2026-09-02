#!/usr/bin/env -S uv run --python 3.11 --script

# /// script
# python = "3.11"
# dependencies = [
#   "onnx<1.17",
#   "onnxruntime",
#   "tf2onnx>=1.16.0",
#   "tensorflow>=2.16",
#   "numpy",
#   "torch",
# ]
# ///

"""
Download and prepare the MediaPipe Face Detector (BlazeFace Short Range)
for testing with burn-onnx.

The model is originally a TensorFlow Lite model that we convert to ONNX
using tf2onnx. This model was the subject of burn issue #1370 regarding
asymmetric padding support in convolutions.

See: https://github.com/tracel-ai/burn/issues/1370
"""

import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_artifacts_dir

# MediaPipe BlazeFace Short Range model from Google's storage
TFLITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)


def download_tflite(output_path):
    """Download the TFLite model from Google storage."""
    print(f"Downloading BlazeFace Short Range TFLite model...")
    print(f"  URL: {TFLITE_URL}")
    urllib.request.urlretrieve(TFLITE_URL, output_path)
    file_size = output_path.stat().st_size / 1024
    print(f"  Downloaded: {file_size:.1f} KB")


def convert_to_onnx(tflite_path, onnx_path, target_opset=16):
    """Convert TFLite model to ONNX using tf2onnx."""
    import tf2onnx

    print(f"Converting TFLite to ONNX (opset {target_opset})...")
    model, _ = tf2onnx.convert.from_tflite(
        str(tflite_path),
        opset=target_opset,
    )

    # Apply shape inference
    print("Applying shape inference...")
    model = shape_inference.infer_shapes(model)

    onnx.save(model, str(onnx_path))
    file_size = onnx_path.stat().st_size / 1024
    print(f"  Saved ONNX model: {file_size:.1f} KB")


def generate_test_data(model_path, output_path):
    """Generate test input/output data and save as PyTorch tensors."""
    import onnxruntime as ort
    import torch

    print("Generating test data...")

    # Load model to inspect inputs/outputs
    session = ort.InferenceSession(str(model_path))

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"  Model inputs:")
    for inp in inputs:
        print(f"    {inp.name}: {inp.shape} ({inp.type})")
    print(f"  Model outputs:")
    for out in outputs:
        print(f"    {out.name}: {out.shape} ({out.type})")

    # Create reproducible test input
    np.random.seed(42)
    input_info = inputs[0]
    # Replace dynamic dims with concrete values
    input_shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
    test_input = np.random.rand(*input_shape).astype(np.float32)

    # Run inference to get reference outputs
    feed = {input_info.name: test_input}
    results = session.run(None, feed)

    # Save as PyTorch tensors with semantic names matching the ONNX outputs
    output_names = [out.name for out in outputs]
    test_data = {"input": torch.from_numpy(test_input)}
    for name, result in zip(output_names, results):
        test_data[name] = torch.from_numpy(result)
        print(f"  {name} shape: {result.shape}")

    torch.save(test_data, output_path)
    print(f"  Test data saved to: {output_path}")


def main():
    print("=" * 60)
    print("MediaPipe Face Detector (BlazeFace Short Range)")
    print("=" * 60)
    print()

    artifacts_dir = get_artifacts_dir("mediapipe-face-detector")

    tflite_path = artifacts_dir / "blaze_face_short_range.tflite"
    onnx_path = artifacts_dir / "face_detector.onnx"
    test_data_path = artifacts_dir / "face_detector_test_data.pt"

    # Check if we already have everything
    if onnx_path.exists() and test_data_path.exists():
        print(f"All files already exist:")
        print(f"  Model: {onnx_path}")
        print(f"  Test data: {test_data_path}")
        print("\nTo re-download, delete the artifacts directory and run again.")
        return

    # Step 1: Download TFLite model
    if not tflite_path.exists():
        print("Step 1: Downloading TFLite model...")
        download_tflite(tflite_path)
    else:
        print(f"Step 1: TFLite model already exists at {tflite_path}")
    print()

    # Step 2: Convert to ONNX
    if not onnx_path.exists():
        print("Step 2: Converting to ONNX...")
        convert_to_onnx(tflite_path, onnx_path)

        # Clean up TFLite file
        tflite_path.unlink()
        print("  Cleaned up TFLite file")
    else:
        print(f"Step 2: ONNX model already exists at {onnx_path}")
    print()

    # Step 3: Generate test data
    if not test_data_path.exists():
        print("Step 3: Generating test data...")
        generate_test_data(onnx_path, test_data_path)
    else:
        print(f"Step 3: Test data already exists at {test_data_path}")
    print()

    print("=" * 60)
    print("Model preparation completed!")
    print("=" * 60)
    print()
    print("Related issue: https://github.com/tracel-ai/burn/issues/1370")
    print()
    print("Next steps:")
    print("  1. Build the model: cargo build")
    print("  2. Run the test:    cargo run")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
