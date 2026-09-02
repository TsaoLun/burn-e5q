#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "numpy==2.2.4",
#   "onnx==1.19.0",
# ]
# ///

# Generates ONNX models and expected outputs for MeanVarianceNormalization tests.
#
# Four configurations are produced:
#   * default axes ([0, 2, 3]): normalize across batch + spatial, per-channel
#   * custom axes ([1, 2]): non-default axes code path
#   * all axes ([0, 1, 2, 3]): reduces to a scalar mean, exercises full broadcast
#   * negative axes ([-4, -2, -1]): equivalent to [0, 2, 3] after resolution

import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model(axes, suffix, input_shape=(2, 3, 4, 5)):
    np.random.seed(42)
    test_input = np.random.randn(*input_shape).astype(np.float32)

    if axes is None:
        node = helper.make_node(
            "MeanVarianceNormalization",
            inputs=["input"],
            outputs=["output"],
        )
    else:
        node = helper.make_node(
            "MeanVarianceNormalization",
            inputs=["input"],
            outputs=["output"],
            axes=axes,
        )

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, list(input_shape)
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, list(input_shape)
    )

    graph = helper.make_graph(
        [node], "mean_variance_normalization_graph", [input_info], [output_info]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.checker.check_model(model)

    file_name = f"mean_variance_normalization_{suffix}.onnx"
    onnx.save(model, file_name)
    print(f"Finished exporting model to {file_name}")

    session = ReferenceEvaluator(model)
    output = session.run(None, {"input": test_input})[0]

    print(f"Test input shape: {test_input.shape}")
    print("Test input:")
    print(np.array2string(test_input, precision=8, max_line_width=120))
    print(f"Test output shape: {output.shape}")
    print("Test output:")
    print(np.array2string(output, precision=8, max_line_width=120))


if __name__ == "__main__":
    build_model(axes=None, suffix="default_axes")
    build_model(axes=[1, 2], suffix="custom_axes")
    build_model(axes=[0, 1, 2, 3], suffix="all_axes")
    build_model(axes=[-4, -2, -1], suffix="negative_axes")
