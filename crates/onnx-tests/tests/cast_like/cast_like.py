#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: cast_like.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto

OPSET_VERSION = 15


def main():
    # CastLike: cast float tensor to match the type of an int tensor
    node0 = helper.make_node(
        "CastLike",
        inputs=["float_input", "int_target"],
        outputs=["float_to_int"],
    )
    # CastLike: cast int tensor to match the type of a float tensor
    node1 = helper.make_node(
        "CastLike",
        inputs=["int_input", "float_target"],
        outputs=["int_to_float"],
    )

    float_input = helper.make_tensor_value_info("float_input", TensorProto.FLOAT, [2, 3])
    int_target = helper.make_tensor_value_info("int_target", TensorProto.INT64, [2, 3])
    int_input = helper.make_tensor_value_info("int_input", TensorProto.INT64, [2, 3])
    float_target = helper.make_tensor_value_info("float_target", TensorProto.FLOAT, [2, 3])

    float_to_int = helper.make_tensor_value_info("float_to_int", TensorProto.INT64, [2, 3])
    int_to_float = helper.make_tensor_value_info("int_to_float", TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph(
        [node0, node1],
        "cast_like_graph",
        [float_input, int_target, int_input, float_target],
        [float_to_int, int_to_float],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, "cast_like.onnx")
    print("Finished exporting model to cast_like.onnx")

    # Verify with ReferenceEvaluator
    from onnx.reference import ReferenceEvaluator

    np.random.seed(42)
    float_data = np.array([[1.0, 2.5, 3.7], [4.1, 5.9, 6.0]], dtype=np.float32)
    int_target_data = np.zeros((2, 3), dtype=np.int64)
    int_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    float_target_data = np.zeros((2, 3), dtype=np.float32)

    sess = ReferenceEvaluator(model)
    results = sess.run(None, {
        "float_input": float_data,
        "int_target": int_target_data,
        "int_input": int_data,
        "float_target": float_target_data,
    })

    print(f"float_to_int: {results[0]} dtype={results[0].dtype}")
    print(f"int_to_float: {results[1]} dtype={results[1].dtype}")


if __name__ == "__main__":
    main()
