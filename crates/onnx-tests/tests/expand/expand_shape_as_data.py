#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: expand_shape_as_data.onnx
#
# Tests Expand where a Shape node's output is the DATA input (input[0]),
# not the target shape (input[1]). This pattern occurs in piper-tts/VITS
# and triggers issue #266 because ArgType::Shape was not handled for
# Expand's data input.
#
# Graph:
#   input: float [3, 4]
#   Shape(input) -> [2] (1D int64: [3, 4])
#   Expand(shape_output, [2, 2]) -> output [2, 2] (int64)

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # Shape(input) -> 1D int64 tensor with the shape values
    shape_node = helper.make_node("Shape", ["input"], ["shape_out"])

    # Target shape [2, 2] as a constant
    target_shape = numpy_helper.from_array(
        np.array([2, 2], dtype=np.int64), name="target_shape"
    )
    target_const = helper.make_node(
        "Constant", [], ["target"], value=target_shape
    )

    # Expand(shape_out, target) -> broadcast shape values to [2, 2]
    expand_node = helper.make_node("Expand", ["shape_out", "target"], ["output"])

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.INT64, [2, 2])

    graph = helper.make_graph(
        [shape_node, target_const, expand_node],
        "expand_shape_as_data",
        [input_info],
        [output_info],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx_name = "expand_shape_as_data.onnx"
    onnx.save(model, onnx_name)
    print(f"Model exported to {onnx_name}")

    # Verify with reference evaluator
    data = np.zeros([3, 4], dtype=np.float32)
    session = ReferenceEvaluator(onnx_name)
    outputs = session.run(None, {"input": data})
    print(f"Input shape: {data.shape}")
    print(f"Output: {outputs[0]} (shape {outputs[0].shape})")
    expected = np.array([[3, 4], [3, 4]], dtype=np.int64)
    np.testing.assert_array_equal(outputs[0], expected)
    print("Test passed!")


if __name__ == "__main__":
    main()
