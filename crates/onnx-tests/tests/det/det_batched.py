#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.17.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/det/det_batched.onnx
#
# Det computes the determinant of a batch of square matrices.
# Input shape: [2, 3, 3] (two 3x3 matrices)
# Output shape: [2] (one determinant per matrix)

import numpy as np
import onnx
from onnx import TensorProto, helper


def main():
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 3])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])

    det_node = helper.make_node(
        "Det",
        inputs=["input"],
        outputs=["output"],
        name="det_node",
    )

    graph = helper.make_graph(
        [det_node],
        "det_batched_model",
        [X],
        [Y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7

    onnx.checker.check_model(model)
    onnx.save(model, "det_batched.onnx")
    print("Finished exporting model to det_batched.onnx")

    # Two 3x3 matrices:
    # Matrix 0: [[1,2,3],[0,1,4],[5,6,0]] -> det = 1*(0-24) - 2*(0-20) + 3*(0-5) = -24+40-15 = 1
    # Matrix 1: [[2,0,0],[0,3,0],[0,0,4]] -> det = 24 (diagonal)
    test_input = np.array(
        [
            [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        ],
        dtype=np.float32,
    )
    print(f"Test input data: {test_input.tolist()}")

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator("det_batched.onnx")
    result = session.run(None, {"input": test_input})
    print(f"Test output data: {result[0]}")  # Expected: [1.0, 24.0]


if __name__ == "__main__":
    main()
