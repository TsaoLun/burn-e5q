#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: maxpool2d_same_upper_dynamic.onnx
#
# MaxPool with auto_pad=SAME_UPPER over dynamic H/W. kernel=[2, 2] with stride=[2, 2] on an
# odd extent leaves a total padding of 1, which SAME_UPPER puts at the end: (0, 1). The windows
# over an extent of 5 are [0,1] [2,3] [4,pad]; SAME_LOWER would shift them to [pad,0] [1,2] [3,4].

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, "H", "W"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, "H_out", "W_out"])

    max_pool = helper.make_node(
        "MaxPool",
        inputs=["x"],
        outputs=["y"],
        kernel_shape=[2, 2],
        strides=[2, 2],
        auto_pad="SAME_UPPER",
    )

    graph = helper.make_graph([max_pool], "maxpool2d_same_upper_dynamic", [x], [y])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    file_name = "maxpool2d_same_upper_dynamic.onnx"
    onnx.save(model, file_name)
    print("Finished exporting model to {}".format(file_name))

    session = ReferenceEvaluator(file_name)
    test_input = np.arange(1, 51, dtype=np.float32).reshape(1, 2, 5, 5)
    output = session.run(None, {"x": test_input})[0]
    print("input {} -> output {}".format(test_input.shape, output.shape))
    print(repr(output))


if __name__ == "__main__":
    main()
