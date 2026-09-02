#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/pad/pad_runtime_pads_axes.onnx
# Tests Pad (opset 18+) with runtime `pads` plus a static `axes` subset.
# Covers the codegen scatter path that places runtime pad pairs onto the
# axes listed by `axes`, leaving (0, 0) on unlisted dimensions.

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
    make_opsetid,
    make_tensor_value_info,
)


def main() -> None:
    # 4D input; pads cover only axes 2 and 0 (in that order).
    inputs = [
        make_tensor_value_info(
            "input_tensor", TensorProto.FLOAT, [None, None, None, None]
        ),
        make_tensor_value_info("pads", TensorProto.INT64, [4]),
    ]
    outputs = [
        make_tensor_value_info(
            "output", TensorProto.FLOAT, [None, None, None, None]
        )
    ]

    axes = numpy_helper.from_array(np.array([2, 0]).astype(np.int64), name="axes")

    node = make_node(
        "Pad",
        inputs=["input_tensor", "pads", "", "axes"],
        outputs=["output"],
        mode="constant",
    )

    graph = make_graph(
        nodes=[node],
        name="PadRuntimePadsAxesGraph",
        inputs=inputs,
        outputs=outputs,
        initializer=[axes],
    )

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)])
    check_model(onnx_model)

    verify(onnx_model)

    onnx.save(onnx_model, Path(__file__).with_name("pad_runtime_pads_axes.onnx"))
    print("Generated pad_runtime_pads_axes.onnx")


def verify(model) -> None:
    sess = ReferenceEvaluator(model)

    input_tensor = np.arange(1, 9, dtype=np.float32).reshape(2, 1, 2, 2)
    # pads layout: [before_axis2, before_axis0, after_axis2, after_axis0]
    pads = np.array([1, 1, 2, 0], dtype=np.int64)

    result = sess.run(None, {"input_tensor": input_tensor, "pads": pads})[0]

    # Expected shape: (2+1+0, 1, 2+1+2, 2) = (3, 1, 5, 2)
    assert result.shape == (3, 1, 5, 2), f"unexpected shape {result.shape}"

    # Spot-check: axis 1 and 3 unpadded; axis 0 gets 1 row before; axis 2
    # gets 1 row before and 2 after.
    expected_zeros_first_batch = np.zeros((1, 5, 2), dtype=np.float32)
    if not np.allclose(result[0], expected_zeros_first_batch):
        print(f"Batch 0 should be zero, got:\n{result[0]}")
        raise Exception("Runtime pads + axes test failed (batch 0)")

    print("Runtime pads + axes test passed!")


if __name__ == "__main__":
    main()
