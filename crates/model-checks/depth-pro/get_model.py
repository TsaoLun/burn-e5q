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

import shutil
import sys
from pathlib import Path

import numpy as np
import onnx

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_artifacts_dir

MODEL_NAME = "depth-pro"
REPO_ID = "onnx-community/DepthPro-ONNX"

# DepthPro default input size from the HuggingFace image processor
INPUT_H = 1536
INPUT_W = 1536


def download_model(artifacts_dir):
    """Download DepthPro ONNX model from Hugging Face.

    The model uses external data format (model.onnx + model.onnx_data).
    Both files are copied to the artifacts directory so they sit side-by-side.
    """
    print("Downloading DepthPro ONNX model from Hugging Face...")
    cache_dir = str(artifacts_dir / "hf_cache")

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="onnx/model.onnx",
        cache_dir=cache_dir,
    )
    print(f"  Downloaded model.onnx")

    data_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="onnx/model.onnx_data",
        cache_dir=cache_dir,
    )
    print(f"  Downloaded model.onnx_data")

    # Copy to artifacts directory. The ONNX file internally references
    # "model.onnx_data" so that name must be preserved.
    onnx_dest = artifacts_dir / f"{MODEL_NAME}.onnx"
    data_dest = artifacts_dir / "model.onnx_data"

    shutil.copy2(model_path, onnx_dest)
    shutil.copy2(data_path, data_dest)
    print(f"  Copied to: {onnx_dest}")
    print(f"  Copied to: {data_dest}")

    return onnx_dest


def get_input_info(model):
    """Extract input information from ONNX model.

    For dynamic dimensions, uses the ONNX dim_param name (e.g. "batch_size",
    "height") to pick a concrete value from DYNAMIC_DIM_DEFAULTS. Falls back
    to 1 for unrecognized names.
    """
    # Map dim_param names/patterns to concrete values for test data generation
    dynamic_dim_defaults = {
        "batch_size": 1, "batch": 1,
        "height": INPUT_H, "width": INPUT_W,
        "channels": 3, "num_channels": 3,
    }

    inputs = []
    for input_info in model.graph.input:
        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                param = dim.dim_param.lower()
                resolved = dynamic_dim_defaults.get(param)
                if resolved is None:
                    # Try substring matching for names like "pixel_values_height"
                    for key, val in dynamic_dim_defaults.items():
                        if key in param:
                            resolved = val
                            break
                if resolved is None:
                    print(f"  Warning: unknown dynamic dim '{dim.dim_param}', defaulting to 1")
                    resolved = 1
                shape.append(resolved)
            else:
                print(f"  Warning: dim has neither dim_value nor dim_param, defaulting to 1")
                shape.append(1)
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

    # Load graph structure only (no weights needed for input/output info)
    model = onnx.load(str(model_path), load_external_data=False)
    input_infos = get_input_info(model)

    print(f"  Model has {len(input_infos)} inputs:")
    for info in input_infos:
        print(f"    - {info['name']}: shape={info['shape']}, dtype={info['dtype']}")

    np.random.seed(42)
    test_inputs = {}

    for info in input_infos:
        test_input = np.random.rand(*info["shape"]).astype(np.float32)
        test_inputs[info["name"]] = test_input

    # onnxruntime handles external data automatically
    session = ort.InferenceSession(str(model_path))
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

    model = onnx.load(str(model_path), load_external_data=False)

    info_path = Path(output_dir) / "model-python.txt"
    with open(info_path, "w") as f:
        f.write("Apple Depth Pro Model Information\n")
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
    print("Apple Depth Pro Model Preparation Tool")
    print("=" * 60)

    artifacts_dir = get_artifacts_dir(MODEL_NAME)

    onnx_path = artifacts_dir / f"{MODEL_NAME}.onnx"
    test_data_path = artifacts_dir / "test_data.pt"
    model_info_path = artifacts_dir / "model-python.txt"

    if onnx_path.exists() and test_data_path.exists() and model_info_path.exists():
        print(f"\n  All files already exist:")
        print(f"  Model: {onnx_path}")
        print(f"  Test data: {test_data_path}")
        print(f"  Model info: {model_info_path}")
        print("\nNothing to do!")
        return

    if not onnx_path.exists():
        print("\nStep 1: Downloading model...")
        download_model(artifacts_dir)

    if not test_data_path.exists():
        print("\nStep 2: Generating test data...")
        generate_test_data(onnx_path, artifacts_dir)

    if not model_info_path.exists():
        print("\nStep 3: Saving model information...")
        save_model_info(onnx_path, artifacts_dir)

    print("\n" + "=" * 60)
    print("  Apple Depth Pro model preparation completed!")
    print(f"  Model: {onnx_path}")
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
