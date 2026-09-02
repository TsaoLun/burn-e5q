#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "numpy==2.2.4",
#   "onnx==1.19.0",
#   "onnxruntime",
# ]
# ///

# Generates ONNX models and expected outputs for LpNormalization tests.
#
# Four configurations are produced:
#   * default attrs (p=2, axis=-1): L2 norm along the last axis
#   * L1 along axis 1: exercises the p=1 code path
#   * L2 along axis 0: exercises a non-trailing axis
#   * L2 with negative axis: axis=-2 on rank-3 should match axis=1
#
# Uses ONNX Runtime instead of `onnx.reference.ReferenceEvaluator` because
# the reference evaluator (as of onnx==1.19.0) has a bug for p=1: it
# computes `x^p` instead of `|x|^p`, so it divides by signed sum rather
# than by the sum of absolute values. ORT and PyTorch's F.normalize both
# use the correct |x|^p form.

import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
import onnxruntime as ort


def build_model(p, axis, suffix, input_shape=(2, 3, 4)):
    np.random.seed(42)
    test_input = np.random.randn(*input_shape).astype(np.float32)

    attrs = {}
    if p is not None:
        attrs["p"] = p
    if axis is not None:
        attrs["axis"] = axis

    node = helper.make_node(
        "LpNormalization",
        inputs=["input"],
        outputs=["output"],
        **attrs,
    )

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, list(input_shape)
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, list(input_shape)
    )

    graph = helper.make_graph(
        [node], "lp_normalization_graph", [input_info], [output_info]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.checker.check_model(model)

    file_name = f"lp_normalization_{suffix}.onnx"
    onnx.save(model, file_name)
    print(f"Finished exporting model to {file_name}")

    session = ort.InferenceSession(file_name)
    output = session.run(None, {"input": test_input})[0]

    print(f"Test input shape: {test_input.shape}")
    print("Test input:")
    print(np.array2string(test_input, precision=8, max_line_width=120))
    print(f"Test output shape: {output.shape}")
    print("Test output:")
    print(np.array2string(output, precision=8, max_line_width=120))


if __name__ == "__main__":
    build_model(p=None, axis=None, suffix="default")
    build_model(p=1, axis=1, suffix="l1_axis1")
    build_model(p=2, axis=0, suffix="l2_axis0")
    build_model(p=2, axis=-2, suffix="l2_negative_axis")
