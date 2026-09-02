#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: avg_pool2d_same_upper_dynamic.onnx
#
# AveragePool with auto_pad=SAME_UPPER over dynamic H/W. kernel=[2, 2] with stride=[2, 2] on an
# odd extent leaves a total padding of 1, placed at the end: (0, 1).
#
# count_include_pad=1 because burn cannot serve the ONNX default of 0 here: its AvgPool2d
# implements asymmetric padding by zero-padding the tensor and then pooling with no padding, so
# the padded cells land in the divisor either way. That is a pre-existing burn limitation which
# explicit asymmetric pads hit identically, not something auto_pad introduces.
#
# TODO: drop count_include_pad and regenerate once tracel-ai/burn#5450 is fixed, so this covers
# the ONNX default. https://github.com/tracel-ai/burn/issues/5450

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, "H", "W"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, "H_out", "W_out"])

    avg_pool = helper.make_node(
        "AveragePool",
        inputs=["x"],
        outputs=["y"],
        kernel_shape=[2, 2],
        strides=[2, 2],
        auto_pad="SAME_UPPER",
        count_include_pad=1,
    )

    graph = helper.make_graph([avg_pool], "avg_pool2d_same_upper_dynamic", [x], [y])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    file_name = "avg_pool2d_same_upper_dynamic.onnx"
    onnx.save(model, file_name)
    print("Finished exporting model to {}".format(file_name))

    session = ReferenceEvaluator(file_name)
    test_input = np.arange(1, 26, dtype=np.float32).reshape(1, 1, 5, 5)
    output = session.run(None, {"x": test_input})[0]
    print("input {} -> output {}".format(test_input.shape, output.shape))
    print(repr(output))


if __name__ == "__main__":
    main()
