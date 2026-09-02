#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/slice/slice_shape_runtime_bounds_reshape.onnx
#
# Verifies that the rank-1 i64 tensor produced by a runtime-bound Shape slice
# is consumable by a downstream Reshape. The IR comment claims downstream ops
# (Reshape, Concat, Gather) accept a tensor as a shape input; this test checks
# that claim end-to-end for the most common consumer.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # `key` is shape (4, 7, 64). Shape -> Slice with runtime [0:2] yields the
    # rank-1 tensor [4, 7]. We then reshape `flat` (28 elements) to that.
    key = helper.make_tensor_value_info("key", TensorProto.FLOAT, [None, None, 64])
    flat = helper.make_tensor_value_info("flat", TensorProto.FLOAT, [28])
    start_in = helper.make_tensor_value_info("start_in", TensorProto.INT64, [1])
    end_in = helper.make_tensor_value_info("end_in", TensorProto.INT64, [1])
    out = helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [None, None])

    nodes = [
        helper.make_node("Shape", inputs=["key"], outputs=["shape_v"]),
        helper.make_node(
            "Slice",
            inputs=["shape_v", "start_in", "end_in"],
            outputs=["sliced_shape"],
        ),
        helper.make_node("Reshape", inputs=["flat", "sliced_shape"], outputs=["reshaped"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="SliceShapeRuntimeBoundsReshape",
        inputs=[key, flat, start_in, end_in],
        outputs=[out],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    onnx.checker.check_model(model)

    onnx_name = "slice_shape_runtime_bounds_reshape.onnx"
    onnx.save(model, onnx_name)
    print(f"Successfully exported model to {onnx_name}")

    sess = ReferenceEvaluator(onnx_name)
    flat_data = np.arange(28, dtype=np.float32)
    out_val, = sess.run(
        None,
        {
            "key": np.zeros((4, 7, 64), dtype=np.float32),
            "flat": flat_data,
            "start_in": np.array([0], dtype=np.int64),
            "end_in": np.array([2], dtype=np.int64),
        },
    )
    print(f"Reference output shape: {out_val.shape}")


if __name__ == "__main__":
    main()
