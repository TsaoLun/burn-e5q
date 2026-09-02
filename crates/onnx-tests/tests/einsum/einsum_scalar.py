#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "torch==2.10.0",
#   "onnxscript",
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generates `crates/onnx-tests/tests/einsum/einsum_scalar.onnx`.

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx.reference import ReferenceEvaluator
from pathlib import Path


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs_scale, matrix, rhs_scale):
        return (
            torch.einsum(",ij->ij", lhs_scale, matrix),
            torch.einsum("ij,->ij", matrix, rhs_scale),
        )


def main():
    model = Model()
    model.eval()
    output_path = Path(__file__).with_suffix(".onnx")

    lhs_scale = torch.tensor(2.0, dtype=torch.float32)
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    rhs_scale = torch.tensor(3.0, dtype=torch.float32)

    torch.onnx.export(
        model,
        (lhs_scale, matrix, rhs_scale),
        output_path,
        verbose=False,
        input_names=["lhs_scale", "matrix", "rhs_scale"],
        output_names=["lhs_scaled", "rhs_scaled"],
        opset_version=16,
        external_data=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    ref = ReferenceEvaluator(onnx_model)
    ref_lhs_scaled, ref_rhs_scaled = ref.run(
        None,
        {
            "lhs_scale": lhs_scale.numpy(),
            "matrix": matrix.numpy(),
            "rhs_scale": rhs_scale.numpy(),
        },
    )
    expected_lhs_scaled, expected_rhs_scaled = model(lhs_scale, matrix, rhs_scale)
    np.testing.assert_allclose(ref_lhs_scaled, expected_lhs_scaled.numpy())
    np.testing.assert_allclose(ref_rhs_scaled, expected_rhs_scaled.numpy())


if __name__ == "__main__":
    main()
