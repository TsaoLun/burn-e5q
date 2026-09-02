#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: size.onnx

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model():
    # Size returns the element count of the input as an INT64 scalar
    input = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 6, 2, 3])
    output = onnx.helper.make_tensor_value_info("output", TensorProto.INT64, [])

    size = onnx.helper.make_node(
        "Size",
        inputs=["input"],
        outputs=["output"],
        name="SizeNode",
    )

    graph = onnx.helper.make_graph(
        [size],
        "SizeModel",
        [input],
        [output],
    )

    model = onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=graph,
        producer_name="ONNX_Generator",
    )

    return model


if __name__ == "__main__":
    np.random.seed(42)
    np.set_printoptions(precision=8)

    onnx_model = build_model()
    file_name = "size.onnx"

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    test_input = np.arange(2 * 6 * 2 * 3, dtype=np.float32).reshape(2, 6, 2, 3)
    print(f"Test input data shape: {test_input.shape}")
    session = ReferenceEvaluator(file_name, verbose=1)
    (test_output,) = session.run(None, {"input": test_input})
    print(f"Test output: {repr(test_output)}")
