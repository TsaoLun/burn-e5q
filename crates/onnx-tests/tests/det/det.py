#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.17.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/det/det.onnx
#
# Det computes the determinant of a square matrix.
# For input [[1, 2], [3, 4]]: det = 1*4 - 2*3 = -2.0

import numpy as np
import onnx
from onnx import TensorProto, helper


def main():
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 2])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [])

    det_node = helper.make_node(
        "Det",
        inputs=["input"],
        outputs=["output"],
        name="det_node",
    )

    graph = helper.make_graph(
        [det_node],
        "det_model",
        [X],
        [Y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7

    onnx.checker.check_model(model)
    onnx.save(model, "det.onnx")
    print("Finished exporting model to det.onnx")

    # Output test data for use in the Rust test
    test_input = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    print(f"Test input data: {test_input.tolist()}")

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator("det.onnx")
    result = session.run(None, {"input": test_input})
    print(f"Test output data: {result[0]}")  # Expected: -2.0


if __name__ == "__main__":
    main()
