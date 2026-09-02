#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: expand_dynamic_where.onnx
#
# Tests Expand with a fully dynamic shape from a Where node (no value_info).
# Reproduces the SDXL UNet pattern where the Expand shape is computed at
# runtime through a Where/ConstantOfShape chain and ONNX shape inference
# cannot determine the shape tensor's length.
#
# Graph:
#   input_data: float [batch]        (1D, dynamic)
#   input_flag: bool  []             (scalar)
#   Shape(input_data) -> [N]         (1D int64, 1 element = batch dim size)
#   Constant([1])     -> [1]         (1D int64, 1 element)
#   Where(input_flag, Shape(input_data), Constant([1])) -> expand_shape
#   Expand(input_data, expand_shape) -> output [batch]

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 14


def main():
    input_data = helper.make_tensor_value_info("input_data", TensorProto.FLOAT, ["batch"])
    input_flag = helper.make_tensor_value_info("input_flag", TensorProto.BOOL, [])

    # Shape(input_data) -> 1D int64 tensor with 1 element (the batch dim size)
    shape_node = helper.make_node("Shape", ["input_data"], ["data_shape"])

    # Constant [1] -> 1D int64 tensor with 1 element
    const_one = helper.make_node(
        "Constant", [], ["one_shape"],
        value=numpy_helper.from_array(np.array([1], dtype=np.int64), name="one"),
    )

    # Where(flag, data_shape, one_shape) -> expand_shape
    # When flag=True: shape = [batch_size] (identity expand)
    # When flag=False: shape = [1] (broadcast to [max(batch,1)] = [batch])
    where_node = helper.make_node(
        "Where", ["input_flag", "data_shape", "one_shape"], ["expand_shape"]
    )

    expand_node = helper.make_node("Expand", ["input_data", "expand_shape"], ["output"])

    # Output shape is dynamic - deliberately no static shape info
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch"])

    graph = helper.make_graph(
        [shape_node, const_one, where_node, expand_node],
        "expand_dynamic_where",
        [input_data, input_flag],
        [output],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx_name = "expand_dynamic_where.onnx"
    onnx.save(model, onnx_name)
    print(f"Model exported to {onnx_name}")

    # Test: flag=True -> expand_shape = [3] -> Expand([10,20,30], [3]) = [10,20,30]
    data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    flag = np.bool_(True)

    session = ReferenceEvaluator(onnx_name)
    outputs = session.run(None, {"input_data": data, "input_flag": flag})
    print(f"Input:  {data} (shape {data.shape})")
    print(f"Output: {outputs[0]} (shape {outputs[0].shape})")
    np.testing.assert_allclose(outputs[0], data)
    print("Test passed!")


if __name__ == "__main__":
    main()
