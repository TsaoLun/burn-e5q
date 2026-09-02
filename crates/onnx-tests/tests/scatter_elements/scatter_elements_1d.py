#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: scatter_elements_1d.onnx

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    # Rank 1 is the degenerate case for the generated coordinate math: the stride loop
    # is empty and the scatter axis is the only coordinate column.
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [4])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [3])
    updates = helper.make_tensor_value_info("updates", TensorProto.FLOAT, [3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])

    node = helper.make_node(
        "ScatterElements",
        inputs=["data", "indices", "updates"],
        outputs=["output"],
        axis=0,
    )

    graph = helper.make_graph(
        [node], "scatter_elements_1d_graph", [data, indices, updates], [output]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "scatter_elements_1d.onnx")

    test_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    test_indices = np.array([2, 0, -1], dtype=np.int64)
    test_updates = np.array([9.0, 8.0, 7.0], dtype=np.float32)

    ref = ReferenceEvaluator(model)
    [result] = ref.run(
        None, {"data": test_data, "indices": test_indices, "updates": test_updates}
    )

    print("Test output: {}".format(result.tolist()))


if __name__ == "__main__":
    main()
