#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "torch==2.10.0",
#   "onnxscript",
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generates `crates/onnx-tests/tests/einsum/einsum_scalar_scalar.onnx`.

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx.reference import ReferenceEvaluator
from pathlib import Path


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs_scale, rhs_scale):
        return torch.einsum(",->", lhs_scale, rhs_scale)


def main():
    model = Model()
    model.eval()
    output_path = Path(__file__).with_suffix(".onnx")

    lhs_scale = torch.tensor(2.0, dtype=torch.float32)
    rhs_scale = torch.tensor(3.0, dtype=torch.float32)

    torch.onnx.export(
        model,
        (lhs_scale, rhs_scale),
        output_path,
        verbose=False,
        input_names=["lhs_scale", "rhs_scale"],
        output_names=["output"],
        opset_version=16,
        external_data=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    ref = ReferenceEvaluator(onnx_model)
    (ref_output,) = ref.run(
        None,
        {
            "lhs_scale": lhs_scale.numpy(),
            "rhs_scale": rhs_scale.numpy(),
        },
    )
    expected_output = model(lhs_scale, rhs_scale)
    np.testing.assert_allclose(ref_output, expected_output.numpy())


if __name__ == "__main__":
    main()
