#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/mel_weight_matrix/mel_weight_matrix_constants.onnx
#
# Mel weight matrix with all 5 inputs baked as initializers (common audio-model pattern).
# num_mel_bins=8, dft_length=16, sample_rate=8192, lower=0, upper=4096.
# Output: [floor(16/2)+1, 8] = [9, 8].

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [9, 8])

    num_mel_bins = numpy_helper.from_array(
        np.array(8, dtype=np.int64), name="num_mel_bins"
    )
    dft_length = numpy_helper.from_array(
        np.array(16, dtype=np.int64), name="dft_length"
    )
    sample_rate = numpy_helper.from_array(
        np.array(8192, dtype=np.int64), name="sample_rate"
    )
    lower_hz = numpy_helper.from_array(
        np.array(0.0, dtype=np.float32), name="lower_edge_hertz"
    )
    upper_hz = numpy_helper.from_array(
        np.array(4096.0, dtype=np.float32), name="upper_edge_hertz"
    )

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
        "mel_weight_matrix_constants_model",
        [],
        [Y],
        initializer=[num_mel_bins, dft_length, sample_rate, lower_hz, upper_hz],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, "mel_weight_matrix_constants.onnx")
    print("Finished exporting model to mel_weight_matrix_constants.onnx")

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator("mel_weight_matrix_constants.onnx")
    result = session.run(None, {})
    print(f"Output shape: {result[0].shape}")
    print(f"Output: {result[0].tolist()}")


if __name__ == "__main__":
    main()
