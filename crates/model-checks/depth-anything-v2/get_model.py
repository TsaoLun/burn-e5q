#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime>=1.22.0",
#   "huggingface-hub>=0.20.0",
#   "numpy",
#   "torch",
# ]
# ///

import sys
from pathlib import Path

import numpy as np
import onnx

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_artifacts_dir, process_model

MODEL_NAME = "depth-anything-v2"


def download_model(output_path):
    """Download Depth-Anything-v2-Small ONNX model from Hugging Face."""
    print("Downloading Depth-Anything-v2-Small model from Hugging Face...")

    model_path = hf_hub_download(
        repo_id="onnx-community/depth-anything-v2-small",
        filename="onnx/model.onnx",
        cache_dir=str(get_artifacts_dir(MODEL_NAME) / "hf_cache"),
    )

    import shutil

    shutil.copy(model_path, output_path)

    if not output_path.exists():
        raise FileNotFoundError(f"Failed to download ONNX file to {output_path}")

    print(f"  Model downloaded to: {output_path}")


def get_input_info(model):
    """Extract input information from ONNX model."""
    inputs = []
    for input_info in model.graph.input:
        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                # Depth-Anything-v2 uses [batch, channels, height, width]
                if len(shape) == 0:
                    shape.append(1)  # batch
                elif len(shape) == 1:
                    shape.append(3)  # channels
                elif len(shape) == 2:
                    shape.append(518)  # height
                elif len(shape) == 3:
                    shape.append(518)  # width
        inputs.append(
            {
                "name": input_info.name,
                "shape": shape,
                "dtype": input_info.type.tensor_type.elem_type,
            }
        )
    return inputs


def generate_test_data(model_path, output_dir):
    """Generate test input/output data and save as PyTorch tensors."""
    import onnxruntime as ort
    import torch

    print("\nGenerating test data...")

    model = onnx.load(model_path)
    input_infos = get_input_info(model)

    print(f"  Model has {len(input_infos)} inputs:")
    for info in input_infos:
        print(f"    - {info['name']}: shape={info['shape']}, dtype={info['dtype']}")

    np.random.seed(42)
    test_inputs = {}

    for info in input_infos:
        test_input = np.random.rand(*info["shape"]).astype(np.float32)
        test_inputs[info["name"]] = test_input

    session = ort.InferenceSession(model_path)
    outputs = session.run(None, test_inputs)

    output_names = [o.name for o in session.get_outputs()]
    print(f"  Model outputs: {output_names}")
    for i, name in enumerate(output_names):
        print(f"    - {name}: shape={outputs[i].shape}")

    test_data = {}

    # Save inputs
    for name, arr in test_inputs.items():
        test_data[name] = torch.from_numpy(arr)

    # Save outputs
    for i, name in enumerate(output_names):
        test_data[name] = torch.from_numpy(outputs[i])

    test_data_path = Path(output_dir) / "test_data.pt"
    torch.save(test_data, test_data_path)

    print(f"  Test data saved to: {test_data_path}")


def save_model_info(model_path, output_dir):
    """Save model structure information to a text file."""
    print("\nSaving model information...")

    model = onnx.load(model_path)

    info_path = Path(output_dir) / "model-python.txt"
    with open(info_path, "w") as f:
        f.write("Depth-Anything-v2-Small Model Information\n")
        f.write("=" * 60 + "\n\n")

        f.write("Inputs:\n")
        for input_info in model.graph.input:
            f.write(f"  - {input_info.name}\n")
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append("dynamic")
            f.write(f"    Shape: {shape}\n")
            f.write(
                f"    Type: {onnx.TensorProto.DataType.Name(input_info.type.tensor_type.elem_type)}\n"
            )

        f.write("\nOutputs:\n")
        for output_info in model.graph.output:
            f.write(f"  - {output_info.name}\n")
            shape = []
            for dim in output_info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append("dynamic")
            f.write(f"    Shape: {shape}\n")
            f.write(
                f"    Type: {onnx.TensorProto.DataType.Name(output_info.type.tensor_type.elem_type)}\n"
            )

        f.write(f"\nModel Statistics:\n")
        f.write(f"  Opset version: {model.opset_import[0].version}\n")
        f.write(f"  Number of nodes: {len(model.graph.node)}\n")
        f.write(f"  Number of initializers: {len(model.graph.initializer)}\n")

        node_types = {}
        for node in model.graph.node:
            node_types[node.op_type] = node_types.get(node.op_type, 0) + 1

        f.write(f"\nNode types:\n")
        for op_type, count in sorted(node_types.items()):
            f.write(f"  {op_type}: {count}\n")

    print(f"  Model info saved to: {info_path}")


def main():
    print("=" * 60)
    print("Depth-Anything-v2-Small Model Preparation Tool")
    print("=" * 60)

    artifacts_dir = get_artifacts_dir(MODEL_NAME)

    original_path = artifacts_dir / f"{MODEL_NAME}.onnx"
    processed_path = artifacts_dir / f"{MODEL_NAME}_opset16.onnx"
    test_data_path = artifacts_dir / "test_data.pt"
    model_info_path = artifacts_dir / "model-python.txt"

    if processed_path.exists() and test_data_path.exists() and model_info_path.exists():
        print(f"\n  All files already exist:")
        print(f"  Model: {processed_path}")
        print(f"  Test data: {test_data_path}")
        print(f"  Model info: {model_info_path}")
        print("\nNothing to do!")
        return

    if not original_path.exists() and not processed_path.exists():
        print("\nStep 1: Downloading model...")
        download_model(original_path)

    if not processed_path.exists():
        print("\nStep 2: Processing model...")
        process_model(original_path, processed_path, target_opset=16)

        if original_path.exists() and processed_path.exists():
            original_path.unlink()

    if not test_data_path.exists():
        print("\nStep 3: Generating test data...")
        generate_test_data(processed_path, artifacts_dir)

    if not model_info_path.exists():
        print("\nStep 4: Saving model information...")
        save_model_info(processed_path, artifacts_dir)

    print("\n" + "=" * 60)
    print("  Depth-Anything-v2-Small model preparation completed!")
    print(f"  Model: {processed_path}")
    print(f"  Test data: {test_data_path}")
    print(f"  Model info: {model_info_path}")
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
