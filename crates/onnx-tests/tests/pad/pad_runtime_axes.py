#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/pad/pad_runtime_axes.onnx
# Tests Pad (opset 18+) with both `pads` and `axes` supplied at runtime,
# including a negative-axis variant exercised in the Rust test.

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator
from onnx.checker import check_model
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_opsetid,
    make_tensor_value_info,
)


def main() -> None:
    inputs = [
        make_tensor_value_info(
            "input_tensor", TensorProto.FLOAT, [None, None, None, None]
        ),
        make_tensor_value_info("pads", TensorProto.INT64, [4]),
        make_tensor_value_info("axes", TensorProto.INT64, [2]),
    ]
    outputs = [
        make_tensor_value_info(
            "output", TensorProto.FLOAT, [None, None, None, None]
        )
    ]

    node = make_node(
        "Pad",
        inputs=["input_tensor", "pads", "", "axes"],
        outputs=["output"],
        mode="constant",
    )

    graph = make_graph(
        nodes=[node],
        name="PadRuntimeAxesGraph",
        inputs=inputs,
        outputs=outputs,
        initializer=[],
    )

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
    check_model(onnx_model)

    verify(onnx_model)

    onnx.save(onnx_model, Path(__file__).with_name("pad_runtime_axes.onnx"))
    print("Generated pad_runtime_axes.onnx")


def verify(model) -> None:
    sess = ReferenceEvaluator(model)
    input_tensor = np.arange(1, 9, dtype=np.float32).reshape(2, 1, 2, 2)
    # axes=[2, 0]: pads layout [before_axis2, before_axis0, after_axis2, after_axis0]
    pads = np.array([1, 1, 2, 0], dtype=np.int64)
    axes = np.array([2, 0], dtype=np.int64)
    result = sess.run(
        None, {"input_tensor": input_tensor, "pads": pads, "axes": axes}
    )[0]
    assert result.shape == (3, 1, 5, 2), f"shape: {result.shape}"

    # Negative axes: -2 == axis 2, -4 == axis 0 for a rank-4 tensor.
    neg_axes = np.array([-2, -4], dtype=np.int64)
    result2 = sess.run(
        None, {"input_tensor": input_tensor, "pads": pads, "axes": neg_axes}
    )[0]
    assert np.array_equal(result, result2), "negative-axis path must agree"
    print("Runtime axes test passed!")


if __name__ == "__main__":
    main()
