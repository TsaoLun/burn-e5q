#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/mel_weight_matrix/mel_weight_matrix_runtime.onnx
#
# Matches the official ONNX test_melweightmatrix layout: all 5 inputs are runtime graph
# inputs (not initializers). This exercises the codegen's runtime path.
# num_mel_bins=8, dft_length=16, sample_rate=8192, lower=0, upper=4096.
# Output: [floor(16/2)+1, 8] = [9, 8].

import numpy as np
import onnx
from onnx import TensorProto, helper


def main():
    num_mel_bins = helper.make_tensor_value_info(
        "num_mel_bins", TensorProto.INT64, []
    )
    dft_length = helper.make_tensor_value_info("dft_length", TensorProto.INT64, [])
    sample_rate = helper.make_tensor_value_info("sample_rate", TensorProto.INT64, [])
    lower_hz = helper.make_tensor_value_info(
        "lower_edge_hertz", TensorProto.FLOAT, []
    )
    upper_hz = helper.make_tensor_value_info(
        "upper_edge_hertz", TensorProto.FLOAT, []
    )
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [9, 8])

    node = helper.make_node(
        "MelWeightMatrix",
        inputs=[
            "num_mel_bins",
            "dft_length",
            "sample_rate",
            "lower_edge_hertz",
            "upper_edge_hertz",
        ],
        outputs=["output"],
        name="mwm_node",
    )

    graph = helper.make_graph(
        [node],
        "mel_weight_matrix_runtime_model",
        [num_mel_bins, dft_length, sample_rate, lower_hz, upper_hz],
        [Y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, "mel_weight_matrix_runtime.onnx")
    print("Finished exporting model to mel_weight_matrix_runtime.onnx")

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator("mel_weight_matrix_runtime.onnx")
    result = session.run(
        None,
        {
            "num_mel_bins": np.int64(8),
            "dft_length": np.int64(16),
            "sample_rate": np.int64(8192),
            "lower_edge_hertz": np.float32(0.0),
            "upper_edge_hertz": np.float32(4096.0),
        },
    )
    print(f"Output shape: {result[0].shape}")
    print(f"Output: {result[0].tolist()}")


if __name__ == "__main__":
    main()
