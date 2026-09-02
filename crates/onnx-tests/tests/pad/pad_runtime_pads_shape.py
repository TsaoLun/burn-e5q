#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/pad/pad_runtime_pads_shape.onnx
# Exercises the path where `pads` is computed from Concat'd Shape outputs.
# burn-onnx's simplification pipeline classifies the result as `Shape(N)`
# rather than a 1D Tensor, which selects a different codegen branch in
# the Pad runtime path (issue #341).

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.checker import check_model
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
)


def main() -> None:
    # The model takes two rank-1 tensors x and y whose lengths encode the
    # pad amounts. Shape(x) is [k_before, k_after_d0] and Shape(y) is
    # [k_before_d1, k_after_d1]; Concat'ing them produces a 4-element
    # pads vector that's typed as Shape(4) after simplification.
    inputs = [
        make_tensor_value_info("data", TensorProto.FLOAT, [None, None]),
        make_tensor_value_info("shape_a", TensorProto.FLOAT, [None, None]),
        make_tensor_value_info("shape_b", TensorProto.FLOAT, [None, None]),
    ]
    outputs = [
        make_tensor_value_info("output", TensorProto.FLOAT, [None, None])
    ]

    nodes = [
        make_node("Shape", inputs=["shape_a"], outputs=["sa"]),
        make_node("Shape", inputs=["shape_b"], outputs=["sb"]),
        make_node("Concat", inputs=["sa", "sb"], outputs=["pads"], axis=0),
        make_node(
            "Pad",
            inputs=["data", "pads"],
            outputs=["output"],
            mode="constant",
        ),
    ]

    graph = make_graph(
        nodes=nodes,
        name="PadRuntimePadsShapeGraph",
        inputs=inputs,
        outputs=outputs,
        initializer=[],
    )

    onnx_model = make_model(graph)
    check_model(onnx_model)

    verify(onnx_model)

    onnx.save(onnx_model, Path(__file__).with_name("pad_runtime_pads_shape.onnx"))
    print("Generated pad_runtime_pads_shape.onnx")


def verify(model) -> None:
    sess = ReferenceEvaluator(model)

    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    # shape_a is (1, 2): contributes pads [1, 2] (before/after for dim 0).
    # shape_b is (2, 1): contributes pads [2, 1] (before/after for dim 1).
    # Final pads = [1, 2, 2, 1] -> ONNX layout [b_d0, b_d1, a_d0, a_d1].
    shape_a = np.zeros((1, 2), dtype=np.float32)
    shape_b = np.zeros((2, 1), dtype=np.float32)

    result = sess.run(
        None, {"data": data, "shape_a": shape_a, "shape_b": shape_b}
    )[0]

    # pads = [1, 2, 2, 1] in ONNX layout [b_d0, b_d1, a_d0, a_d1]
    # → output shape (2+1+2, 2+2+1) = (5, 5)
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    if not np.allclose(result, expected):
        print(f"Expected:\n{expected}\nGot:\n{result}")
        raise Exception("Pad with Shape-typed pads test failed")

    print("Pad with Shape-typed pads test passed!")


if __name__ == "__main__":
    main()
