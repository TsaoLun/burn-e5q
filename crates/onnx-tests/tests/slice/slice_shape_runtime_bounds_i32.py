#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/slice/slice_shape_runtime_bounds_i32.onnx
#
# Sibling of slice_shape_runtime_bounds.py with int32 bounds. ONNX Slice's
# Tind type permits int32 or int64; the codegen does an explicit cast to
# i64, and this test exercises that path end-to-end (the i64 sibling does
# not).

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    key = helper.make_tensor_value_info("key", TensorProto.FLOAT, [None, None, 64])
    start_in = helper.make_tensor_value_info("start_in", TensorProto.INT32, [1])
    end_in = helper.make_tensor_value_info("end_in", TensorProto.INT32, [1])
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
        name="SliceShapeRuntimeBoundsI32",
        inputs=[key, start_in, end_in],
        outputs=[out],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    onnx.checker.check_model(model)

    onnx_name = "slice_shape_runtime_bounds_i32.onnx"
    onnx.save(model, onnx_name)
    print(f"Successfully exported model to {onnx_name}")

    sess = ReferenceEvaluator(onnx_name)
    out_val, = sess.run(
        None,
        {
            "key": np.zeros((4, 7, 64), dtype=np.float32),
            "start_in": np.array([1], dtype=np.int32),
            "end_in": np.array([3], dtype=np.int32),
        },
    )
    print(f"Reference output: {out_val}")


if __name__ == "__main__":
    main()
