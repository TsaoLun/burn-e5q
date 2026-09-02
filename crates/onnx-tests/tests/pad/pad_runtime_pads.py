#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/pad/pad_runtime_pads.onnx
# Tests Pad with the `pads` input supplied at runtime (graph input,
# not a static initializer). Issues #341 and #405.

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
    make_tensor_value_info,
)


def main() -> None:
    inputs = [
        make_tensor_value_info("input_tensor", TensorProto.FLOAT, [None, None]),
        make_tensor_value_info("pads", TensorProto.INT64, [4]),
    ]
    outputs = [make_tensor_value_info("output", TensorProto.FLOAT, [None, None])]

    node = make_node(
        "Pad",
        inputs=["input_tensor", "pads"],
        outputs=["output"],
        mode="constant",
    )

    graph = make_graph(
        nodes=[node],
        name="PadRuntimePadsGraph",
        inputs=inputs,
        outputs=outputs,
        initializer=[],
    )

    onnx_model = make_model(graph)
    check_model(onnx_model)

    verify(onnx_model)

    onnx.save(onnx_model, Path(__file__).with_name("pad_runtime_pads.onnx"))
    print("Generated pad_runtime_pads.onnx")


def verify(model) -> None:
    sess = ReferenceEvaluator(model)

    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pads = np.array([1, 2, 1, 2], dtype=np.int64)

    result = sess.run(None, {"input_tensor": input_tensor, "pads": pads})[0]

    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    if not np.allclose(result, expected):
        print(f"Expected:\n{expected}")
        print(f"Got:\n{result}")
        raise Exception("Runtime pads test failed")

    print("Runtime pads test passed!")


if __name__ == "__main__":
    main()
