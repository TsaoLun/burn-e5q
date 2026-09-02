#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Regression test: Where with tensor condition and both x, y as scalar constants.
# This pattern occurs in ALBERT for attention masking:
#   Where(mask, -inf, 0.0) -> fill -inf where mask is true, 0 elsewhere
#
# The bug: y (scalar) gets converted to a [1,1] tensor, but the condition
# may be [1,1,N,M]. mask_fill requires the mask to broadcast to the tensor
# shape, which fails when the tensor is smaller than the mask.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # Constants
    const_x = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_x"],
        value=helper.make_tensor("x_val", TensorProto.FLOAT, [], [-1.0]),
    )
    const_y = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_y"],
        value=helper.make_tensor("y_val", TensorProto.FLOAT, [], [0.0]),
    )

    # Where(condition, x=-1.0, y=0.0)
    where_node = helper.make_node(
        "Where", ["condition", "const_x", "const_y"], ["output"]
    )

    condition_info = helper.make_tensor_value_info(
        "condition", TensorProto.BOOL, [2, 3]
    )
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph(
        [const_x, const_y, where_node],
        "where_scalar_xy",
        [condition_info],
        [output_info],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.save(model, "where_scalar_xy.onnx")
    print("Saved where_scalar_xy.onnx")

    # Verify with ReferenceEvaluator
    cond = np.array(
        [[True, False, True], [False, True, False]], dtype=np.bool_
    )
    session = ReferenceEvaluator("where_scalar_xy.onnx")
    [result] = session.run(None, {"condition": cond})
    print(f"Condition: {cond}")
    print(f"Output:    {result}")
    expected = np.where(cond, -1.0, 0.0).astype(np.float32)
    assert np.array_equal(result, expected)


if __name__ == "__main__":
    main()
