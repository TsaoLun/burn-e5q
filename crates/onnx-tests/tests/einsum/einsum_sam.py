#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "torch==2.10.0",
#   "onnxscript",
#   "onnx==1.19.0",
# ]
# ///

# Generates `crates/onnx-tests/tests/einsum/einsum_sam.onnx`.

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, r_q, r_h):
        return torch.einsum("bhwc,hkc->bhwk", r_q, r_h)


def main():
    torch.manual_seed(42)

    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "einsum_sam.onnx"

    r_q = torch.randn(1, 2, 3, 4, device=device)
    r_h = torch.randn(2, 5, 4, device=device)

    test_input = (r_q, r_h)

    torch.onnx.export(
        model,
        test_input,
        onnx_name,
        verbose=False,
        opset_version=16,
        external_data=False,
    )

    print(f"Finished exporting model to {onnx_name}")
    print(f"Test input shapes: r_q={r_q.shape}, r_h={r_h.shape}")
    output = model.forward(*test_input)
    print(f"Test output shape: {output.shape}")
    print(f"Test output: {output}")


if __name__ == "__main__":
    main()
