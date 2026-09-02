#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "numpy==2.2.4",
#   "onnx==1.19.0",
#   "onnxruntime",
# ]
# ///


import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
import onnxruntime as ort


def build_model(p, suffix, input_shape=(2, 3, 4), opset=7):
    np.random.seed(42)

    attrs = {}
    if p is not None:
        attrs["p"] = p

    node = helper.make_node(
        "GlobalLpPool",
        inputs=["input"],
        outputs=["output"],
        **attrs,
    )

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, list(input_shape)
    )
    # N and C carry through; every spatial dim collapses to 1.
    output_shape = list(input_shape[:2]) + [1] * (len(input_shape) - 2)

    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )

    graph = helper.make_graph(
        [node], "global_lp_pooling_graph", [input_info], [output_info]
    )
    # Opset 7 keeps `p` as an INT attribute; opset 1 declares it FLOAT, which is the
    # only encoding that can carry a fractional p. Parsing across every opset the
    # operator exists in is covered by the opset-compliance harness instead.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8

    onnx.checker.check_model(model)

    file_name = f"global_lp_pool_{suffix}.onnx"
    onnx.save(model, file_name)
    print(f"Finished exporting model to {file_name}")

    test_input = np.random.randn(*input_shape).astype(np.float32)
    # onnx.reference.ReferenceEvaluator has no GlobalLpPool implementation at all
    # (RuntimeImplementationError), and onnxruntime has no GlobalLpPool(1) kernel
    # (NOT_IMPLEMENTED), so opset 1 falls back to the formula from the spec.
    if opset >= 2:
        session = ort.InferenceSession(file_name, providers=["CPUExecutionProvider"])
        output = session.run(None, {"input": test_input})[0]
    else:
        norm = 2.0 if p is None else float(p)
        axes = tuple(range(2, len(input_shape)))
        output = (
            (np.abs(test_input.astype(np.float64)) ** norm).sum(axis=axes, keepdims=True)
            ** (1.0 / norm)
        ).astype(np.float32)

    print(f"Test input shape: {test_input.shape}")
    print("Test input:")
    print(np.array2string(test_input, precision=8, max_line_width=120))
    print(f"Test output shape: {output.shape}")
    print("Test output:")
    print(np.array2string(output, precision=8, max_line_width=120))


if __name__ == "__main__":
    build_model(p=None, suffix="default")
    build_model(p=1, suffix="l1")
    build_model(p=2, suffix="l2")
    build_model(p=3, suffix="l3")
    build_model(p=1, suffix="rank_4_l1", input_shape=(2, 3, 2, 3))
    build_model(p=2, suffix="rank_4_l2", input_shape=(2, 3, 2, 3))
    build_model(p=3, suffix="rank_4_l3", input_shape=(2, 3, 2, 3))
    # Opset 1 declares `p` as FLOAT and puts no integrality constraint on it.
    build_model(p=2.5, suffix="opset1_fractional_p", opset=1)
