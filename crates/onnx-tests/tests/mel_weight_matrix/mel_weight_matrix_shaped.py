#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/mel_weight_matrix/mel_weight_matrix_shaped.onnx
#
# Exercises non-degenerate triangles (filters with real slopes, not just 1.0 peaks).
# num_mel_bins=4, dft_length=32 (n_bins=17), sample_rate=16000, 300-4000 Hz range.

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
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [17, 4])

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
        "mel_weight_matrix_shaped_model",
        [num_mel_bins, dft_length, sample_rate, lower_hz, upper_hz],
        [Y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, "mel_weight_matrix_shaped.onnx")
    print("Finished exporting model to mel_weight_matrix_shaped.onnx")

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator("mel_weight_matrix_shaped.onnx")
    result = session.run(
        None,
        {
            "num_mel_bins": np.int64(4),
            "dft_length": np.int64(32),
            "sample_rate": np.int64(16000),
            "lower_edge_hertz": np.float32(300.0),
            "upper_edge_hertz": np.float32(4000.0),
        },
    )
    print(f"Output shape: {result[0].shape}")
    print(f"Output: {result[0].tolist()}")


if __name__ == "__main__":
    main()
