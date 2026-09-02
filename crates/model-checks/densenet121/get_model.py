#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime>=1.22.0",
#   "onnxscript",
#   "numpy",
#   "torch",
#   "torchvision",
# ]
# ///

import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torchvision.models as models
from onnx import shape_inference

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_artifacts_dir

MODEL_NAME = "densenet121"


def export_model(output_path):
    """Export DenseNet-121 to ONNX format."""
    print("Loading DenseNet-121 from torchvision...")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Exporting to ONNX at {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        opset_version=18,
    )

    print("Applying shape inference...")
    m = onnx.load(output_path)
    m = shape_inference.infer_shapes(m)
    onnx.save(m, output_path)
    print(f"  Model exported to: {output_path}")


def generate_test_data(model_path, output_dir):
    """Generate test input/output data using ONNX Runtime."""
    import onnxruntime as ort

    print("\nGenerating test data...")

    np.random.seed(42)
    test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

    session = ort.InferenceSession(model_path)
    outputs = session.run(None, {"input": test_input})

    output_names = [o.name for o in session.get_outputs()]
    print(f"  Model outputs: {output_names}")
    for i, name in enumerate(output_names):
        print(f"    - {name}: shape={outputs[i].shape}")

    test_data = {"input": torch.from_numpy(test_input)}
    for i, name in enumerate(output_names):
        test_data[name] = torch.from_numpy(outputs[i])

    test_data_path = Path(output_dir) / "test_data.pt"
    torch.save(test_data, test_data_path)
    print(f"  Test data saved to: {test_data_path}")


def main():
    print("=" * 60)
    print("DenseNet-121 Model Preparation")
    print("=" * 60)

    artifacts_dir = get_artifacts_dir(MODEL_NAME)

    model_path = artifacts_dir / f"{MODEL_NAME}.onnx"
    test_data_path = artifacts_dir / "test_data.pt"

    if model_path.exists() and test_data_path.exists():
        print(f"\n  All files already exist:")
        print(f"  Model: {model_path}")
        print(f"  Test data: {test_data_path}")
        print("\nNothing to do!")
        return

    if not model_path.exists():
        print("\nStep 1: Exporting model...")
        export_model(model_path)

    if not test_data_path.exists():
        print("\nStep 2: Generating test data...")
        generate_test_data(model_path, artifacts_dir)

    print("\n" + "=" * 60)
    print(f"  DenseNet-121 model preparation completed!")
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
