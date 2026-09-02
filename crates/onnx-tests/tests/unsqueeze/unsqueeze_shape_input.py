#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: unsqueeze_shape_input.onnx
#
# Reproduces issue #258: Unsqueeze fails when data input is a Shape type.
#
# Pattern: Shape -> Gather (scalar index) -> Unsqueeze -> Concat -> Reshape
# The Shape op outputs ArgType::Shape in onnx-ir, and when this feeds into
# operations that don't handle Shape, the pipeline fails.
#
# This test uses a simpler pattern: Shape -> Unsqueeze directly.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator


def build_model():
    # Input: a 2D float tensor
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])

    # Output: the shape of X, unsqueezed to 2D (1 x ndim)
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [1, 2])

    # Shape(X) -> 1D int64 tensor [2, 3]
    shape_node = helper.make_node("Shape", inputs=["X"], outputs=["shape_out"])

    # Axes constant for unsqueeze: add dimension at axis 0
    axes_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["axes"],
        value=helper.make_tensor(
            name="axes_value", data_type=TensorProto.INT64, dims=[1], vals=[0]
        ),
    )

    # Unsqueeze(shape_out, axes=[0]) -> 2D tensor [[2, 3]]
    unsqueeze_node = helper.make_node(
        "Unsqueeze", inputs=["shape_out", "axes"], outputs=["Y"]
    )

    graph = helper.make_graph(
        [shape_node, axes_const, unsqueeze_node],
        "unsqueeze_shape_input",
        [X],
        [Y],
    )

    model = helper.make_model(
        graph,
        producer_name="unsqueeze_shape_input_test",
        opset_imports=[helper.make_operatorsetid("", 16)],
    )
    model.ir_version = 8

    return model


def main():
    model = build_model()
    onnx.save(model, "unsqueeze_shape_input.onnx")
    print("Saved unsqueeze_shape_input.onnx")

    onnx.checker.check_model(model)

    session = ReferenceEvaluator(model, verbose=0)

    np.random.seed(42)
    test_input = np.random.randn(2, 3).astype(np.float32)
    print(f"Input shape: {test_input.shape}")

    (output,) = session.run(None, {"X": test_input})
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")

    expected = np.array([[2, 3]], dtype=np.int64)
    assert np.array_equal(output, expected), f"Expected {expected}, got {output}"
    print("Test passed!")


if __name__ == "__main__":
    main()
