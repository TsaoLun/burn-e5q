#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "torch==2.10.0",
#   "onnxscript",
#   "onnx==1.19.0",
# ]
# ///

# Generates `crates/onnx-tests/tests/einsum/einsum_shadow_rhs.onnx`.

import torch
import torch.nn as nn
from pathlib import Path


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs, einsum_rhs):
        product = torch.einsum("ij,kj->ik", lhs, einsum_rhs)
        reused_input = einsum_rhs.sum(dim=1).unsqueeze(0)
        return product + reused_input


def main():
    model = Model()
    model.eval()
    output_path = Path(__file__).with_suffix(".onnx")

    lhs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    einsum_rhs = torch.tensor(
        [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]],
        dtype=torch.float32,
    )

    torch.onnx.export(
        model,
        (lhs, einsum_rhs),
        output_path,
        verbose=False,
        input_names=["lhs", "einsum_rhs"],
        output_names=["output"],
        opset_version=16,
        external_data=False,
    )


if __name__ == "__main__":
    main()
