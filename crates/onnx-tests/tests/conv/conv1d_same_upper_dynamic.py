#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
#   "onnxruntime",
# ]
# ///

# used to generate model: conv1d_same_upper_dynamic.onnx
#
# Conv with auto_pad=SAME_UPPER over a dynamic length. This is the 1D counterpart of
# conv2d_same_upper_dynamic and exists so PaddingConfig1d::Same is actually compiled and run,
# not just asserted as generated text.
#
# strides=[2] makes the pads depend on the parity of the length:
#   L=9 -> (1, 2), the asymmetric case SAME_UPPER puts last
#   L=8 -> (1, 1)
#
# Ground truth comes from onnxruntime; see conv2d_same_upper_dynamic.py for why not
# onnx.reference.ReferenceEvaluator.

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnxruntime import InferenceSession


def main():
    np.random.seed(42)

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, "L"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "L_out"])

    weight = numpy_helper.from_array(np.random.randn(3, 2, 4).astype(np.float32), "weight")
    bias = numpy_helper.from_array(np.random.randn(3).astype(np.float32), "bias")

    conv = helper.make_node(
        "Conv",
        inputs=["x", "weight", "bias"],
        outputs=["y"],
        kernel_shape=[4],
        strides=[2],
        auto_pad="SAME_UPPER",
    )

    graph = helper.make_graph(
        [conv],
        "conv1d_same_upper_dynamic",
        [x],
        [y],
        initializer=[weight, bias],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    file_name = "conv1d_same_upper_dynamic.onnx"
    onnx.save(model, file_name)
    print("Finished exporting model to {}".format(file_name))

    session = InferenceSession(file_name)
    for length in [9, 8]:
        test_input = np.ones((1, 2, length), dtype=np.float32)
        output = session.run(None, {"x": test_input})[0]
        print(
            "input {} -> output {}, sum {:.6f}".format(
                test_input.shape, output.shape, output.sum()
            )
        )


if __name__ == "__main__":
    main()
