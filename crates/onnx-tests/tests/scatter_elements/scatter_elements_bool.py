#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: scatter_elements_bool.onnx

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    data = helper.make_tensor_value_info("data", TensorProto.BOOL, [2, 3])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 3])
    updates = helper.make_tensor_value_info("updates", TensorProto.BOOL, [2, 3])
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [2, 3])

    node = helper.make_node(
        "ScatterElements",
        inputs=["data", "indices", "updates"],
        outputs=["output"],
        axis=1,
    )

    graph = helper.make_graph(
        [node], "scatter_elements_bool_graph", [data, indices, updates], [output]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "scatter_elements_bool.onnx")

    # Covers both directions: some targets are set true, others cleared to false.
    test_data = np.array([[True, True, True], [False, False, False]], dtype=bool)
    test_indices = np.array([[2, 0, 1], [1, 2, 0]], dtype=np.int64)
    test_updates = np.array([[False, False, True], [True, True, False]], dtype=bool)

    ref = ReferenceEvaluator(model)
    [result] = ref.run(
        None, {"data": test_data, "indices": test_indices, "updates": test_updates}
    )

    print("Test data: {}".format(test_data))
    print("Test indices: {}".format(test_indices))
    print("Test updates: {}".format(test_updates))
    print("Test output: {}".format(result))


if __name__ == "__main__":
    main()
