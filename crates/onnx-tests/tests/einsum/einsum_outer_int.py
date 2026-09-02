#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "torch==2.10.0",
#   "onnxscript",
#   "onnx==1.19.0",
# ]
# ///

# Generates `crates/onnx-tests/tests/einsum/einsum_outer_int.onnx`.

import torch
import torch.nn as nn
from pathlib import Path


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.einsum("i,j->ij", a, b)


def main():
    model = Model()
    model.eval()
    output_path = Path(__file__).with_suffix(".onnx")

    a = torch.tensor([1, 2, 3], dtype=torch.int32)
    b = torch.tensor([4, 5, 6], dtype=torch.int32)

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


if __name__ == "__main__":
    main()
