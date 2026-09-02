#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "torch==2.10.0",
#   "onnxscript",
#   "onnx==1.19.0",
# ]
# ///

# Generates `crates/onnx-tests/tests/einsum/einsum.onnx`.

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c, d):
        return (
            torch.einsum("ij,jk->ik", a, b),
            torch.einsum("bij,bjk->bik", c, d),
        )


def main():
    torch.manual_seed(42)

    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "einsum.onnx"

    a = torch.randn(3, 4, device=device)
    b = torch.randn(4, 2, device=device)
    c = torch.randn(2, 3, 4, device=device)
    d = torch.randn(2, 4, 5, device=device)

    test_input = (a, b, c, d)

    torch.onnx.export(
        model,
        test_input,
        onnx_name,
        verbose=False,
        opset_version=16,
        external_data=False,
    )

    print(f"Finished exporting model to {onnx_name}")
    print(f"Test input data shapes: a={a.shape}, b={b.shape}, c={c.shape}, d={d.shape}")
    output = model.forward(*test_input)
    print(f"Test output matmul: {output[0]}")
    print(f"Test output batch_matmul: {output[1]}")


if __name__ == "__main__":
    main()
