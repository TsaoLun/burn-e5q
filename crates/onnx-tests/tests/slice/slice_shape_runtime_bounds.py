#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/slice/slice_shape_runtime_bounds.onnx
#
# `start_in` and `end_in` are graph inputs, so neither bound can be lifted to
# a static SliceInput; the IR must produce a rank-1 i64 tensor and the codegen
# has to materialize it on device.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # Inputs
    key = helper.make_tensor_value_info("key", TensorProto.FLOAT, [None, None, 64])
    start_in = helper.make_tensor_value_info("start_in", TensorProto.INT64, [1])
    end_in = helper.make_tensor_value_info("end_in", TensorProto.INT64, [1])
    out = helper.make_tensor_value_info("sliced_shape", TensorProto.INT64, [None])

    nodes = [
        helper.make_node("Shape", inputs=["key"], outputs=["shape_v"]),
        helper.make_node(
            "Slice",
            inputs=["shape_v", "start_in", "end_in"],
            outputs=["sliced_shape"],
        ),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="SliceShapeRuntimeBounds",
        inputs=[key, start_in, end_in],
        outputs=[out],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    onnx.checker.check_model(model)

    onnx_name = "slice_shape_runtime_bounds.onnx"
    onnx.save(model, onnx_name)
    print(f"Successfully exported model to {onnx_name}")

    # Sanity-check with reference evaluator: key shape (4, 7, 64), slice [0:2]
    sess = ReferenceEvaluator(onnx_name)
    out_val, = sess.run(
        None,
        {
            "key": np.zeros((4, 7, 64), dtype=np.float32),
            "start_in": np.array([0], dtype=np.int64),
            "end_in": np.array([2], dtype=np.int64),
        },
    )
    print(f"Reference output: {out_val}")


if __name__ == "__main__":
    main()
