#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: gather_shape_reorder.onnx
#
# Reproduces part of issue #258: Gather(Shape, tensor_indices) produced
# wrong Shape rank, breaking downstream Slice/Concat operations.
#
# Pattern from bodyposenet: Shape -> Gather([0,2,3,1]) -> Slice -> Concat
# This reorders NCHW shape dims to NHWC, then slices out spatial dims.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator


def build_model():
    # Input: a 4D float tensor (NCHW)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 6])

    # Output: the H and W dimensions extracted after reordering
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [2])

    # Shape(X) -> [1, 3, 8, 6]
    shape_node = helper.make_node("Shape", inputs=["X"], outputs=["shape_out"])

    # Constant indices [0, 2, 3, 1] to reorder NCHW -> NHWC
    indices_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["indices"],
        value=helper.make_tensor(
            name="idx", data_type=TensorProto.INT64, dims=[4], vals=[0, 2, 3, 1]
        ),
    )

    # Gather(shape, [0,2,3,1]) -> reordered shape [1, 8, 6, 3]
    gather_node = helper.make_node(
        "Gather",
        inputs=["shape_out", "indices"],
        outputs=["reordered"],
        axis=0,
    )

    # Slice to extract spatial dims (indices 1 and 2 of reordered = H, W)
    starts_const = helper.make_node(
        "Constant", inputs=[], outputs=["starts"],
        value=helper.make_tensor("s", TensorProto.INT64, [1], [1]),
    )
    ends_const = helper.make_node(
        "Constant", inputs=[], outputs=["ends"],
        value=helper.make_tensor("e", TensorProto.INT64, [1], [3]),
    )
    axes_const = helper.make_node(
        "Constant", inputs=[], outputs=["axes"],
        value=helper.make_tensor("a", TensorProto.INT64, [1], [0]),
    )

    # Slice(reordered, [1], [3], [0]) -> [8, 6]
    slice_node = helper.make_node(
        "Slice",
        inputs=["reordered", "starts", "ends", "axes"],
        outputs=["Y"],
    )

    graph = helper.make_graph(
        [shape_node, indices_const, gather_node, starts_const, ends_const,
         axes_const, slice_node],
        "gather_shape_reorder",
        [X],
        [Y],
    )

    model = helper.make_model(
        graph,
        producer_name="gather_shape_reorder_test",
        opset_imports=[helper.make_operatorsetid("", 16)],
    )
    model.ir_version = 8
    return model


def main():
    model = build_model()
    onnx.save(model, "gather_shape_reorder.onnx")
    print("Saved gather_shape_reorder.onnx")

    onnx.checker.check_model(model)

    session = ReferenceEvaluator(model, verbose=0)

    test_input = np.ones((1, 3, 8, 6), dtype=np.float32)
    (output,) = session.run(None, {"X": test_input})
    print(f"Output: {output}")  # [8, 6]

    expected = np.array([8, 6], dtype=np.int64)
    assert np.array_equal(output, expected), f"Expected {expected}, got {output}"
    print("Test passed!")


if __name__ == "__main__":
    main()
