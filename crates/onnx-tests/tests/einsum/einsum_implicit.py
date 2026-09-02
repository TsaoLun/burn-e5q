#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "torch==2.10.0",
#   "onnxscript",
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generates `crates/onnx-tests/tests/einsum/einsum_implicit.onnx`.
# Tests implicit output form (no -> in equation).

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx.reference import ReferenceEvaluator
from pathlib import Path


class Model(nn.Module):
    def forward(self, a, b):
        # Implicit form: "ij,jk" means j is summed out, output is "ik"
        return torch.einsum("ij,jk", a, b)


def main():
    model = Model()
    model.eval()
    output_path = Path(__file__).with_suffix(".onnx")

    torch.manual_seed(42)
    a = torch.randn(3, 4, dtype=torch.float32)
    b = torch.randn(4, 2, dtype=torch.float32)

    torch.onnx.export(
        model,
        (a, b),
        output_path,
        verbose=False,
        input_names=["a", "b"],
        output_names=["output"],
        opset_version=16,
        external_data=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    ref = ReferenceEvaluator(onnx_model)
    [ref_output] = ref.run(None, {"a": a.numpy(), "b": b.numpy()})
    expected = model(a, b)
    np.testing.assert_allclose(ref_output, expected.numpy(), rtol=1e-5)


if __name__ == "__main__":
    main()
