#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
#   "onnxruntime",
# ]
# ///

# used to generate model: conv2d_same_upper_dynamic.onnx
#
# Conv with auto_pad=SAME_UPPER over dynamic H/W. The pads cannot be computed at import
# time, so they have to be derived from the real input size at forward time.
#
# strides=[2, 2] is what makes this test bite: at stride 1 the total padding is kernel - 1
# whatever the input size, so every input would share one pad. At stride 2 it depends on the
# parity of the extent, and the two sizes below land on different pads:
#   5x6 -> H (1, 1), W (1, 1)
#   8x7 -> H (0, 1), W (1, 2)
# The odd totals also exercise the asymmetric side of SAME_UPPER, which puts the extra pad last.
#
# Ground truth comes from onnxruntime rather than onnx.reference.ReferenceEvaluator: the latter
# gets Conv + SAME_UPPER wrong at stride 2 over an odd extent, returning W_out=3 for W=7 where
# the spec requires ceil(7/2)=4.

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnxruntime import InferenceSession


def main():
    np.random.seed(42)

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, "H", "W"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "H_out", "W_out"])

    weight = numpy_helper.from_array(
        np.random.randn(3, 2, 3, 4).astype(np.float32), "weight"
    )
    bias = numpy_helper.from_array(np.random.randn(3).astype(np.float32), "bias")

    conv = helper.make_node(
        "Conv",
        inputs=["x", "weight", "bias"],
        outputs=["y"],
        kernel_shape=[3, 4],
        strides=[2, 2],
        auto_pad="SAME_UPPER",
    )

    graph = helper.make_graph(
        [conv],
        "conv2d_same_upper_dynamic",
        [x],
        [y],
        initializer=[weight, bias],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    file_name = "conv2d_same_upper_dynamic.onnx"
    onnx.save(model, file_name)
    print("Finished exporting model to {}".format(file_name))

    # Two different spatial sizes through the same model: the padding has to adapt.
    session = InferenceSession(file_name)
    for height, width in [(5, 6), (8, 7)]:
        test_input = np.ones((1, 2, height, width), dtype=np.float32)
        output = session.run(None, {"x": test_input})[0]
        print(
            "input {} -> output {}, sum {:.6f}".format(
                test_input.shape, output.shape, output.sum()
            )
        )


if __name__ == "__main__":
    main()
