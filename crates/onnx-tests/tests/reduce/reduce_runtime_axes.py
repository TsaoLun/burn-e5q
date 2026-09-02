#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: reduce_runtime_axes.onnx
#
# Opset 18 moved `axes` from an attribute to an optional input (opset 13 for ReduceSum).
# When that input is a graph input rather than a constant, its value is unknown at build
# time, which is a different thing from "no axes were given" (tracel-ai/burn-onnx#459).
# The two must not collapse: an absent `axes` reduces every dimension, a runtime `axes`
# reduces only the dimensions it names.
#
# The `noop_with_empty_axes` output covers the third meaning of an empty axes list: with
# the attribute set the reduction is skipped, though for ops with an elementwise part
# (ReduceSumSquare, ReduceL1, ReduceL2, ReduceLogSum) that part still applies.

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 18


def build_model():
    data = onnx.helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 4])
    axes = onnx.helper.make_tensor_value_info("axes", TensorProto.INT64, [1])
    axes2 = onnx.helper.make_tensor_value_info("axes2", TensorProto.INT64, [2])
    empty_axes = onnx.helper.make_tensor_value_info(
        "empty_axes", TensorProto.INT64, [0]
    )

    # keepdims=1: the reduced dimension survives as size 1, so the rank is unchanged.
    sum_keepdims = onnx.helper.make_node(
        "ReduceSum", ["data", "axes"], ["sum_keepdims"], keepdims=1
    )
    # keepdims=0 over two axes: the output rank comes from the length of `axes2`,
    # which is static even though its values are not.
    mean_no_keepdims = onnx.helper.make_node(
        "ReduceMean", ["data", "axes2"], ["mean_no_keepdims"], keepdims=0
    )
    # A negative axis is resolved at run time.
    max_keepdims = onnx.helper.make_node(
        "ReduceMax", ["data", "axes"], ["max_keepdims"], keepdims=1
    )
    # Empty axes with noop_with_empty_axes=1 is an identity, not a full reduction.
    sum_noop = onnx.helper.make_node(
        "ReduceSum",
        ["data", "empty_axes"],
        ["sum_noop"],
        keepdims=1,
        noop_with_empty_axes=1,
    )

    graph = onnx.helper.make_graph(
        [sum_keepdims, mean_no_keepdims, max_keepdims, sum_noop],
        "ReduceRuntimeAxesModel",
        [data, axes, axes2, empty_axes],
        [
            onnx.helper.make_tensor_value_info(
                "sum_keepdims", TensorProto.FLOAT, [2, 1, 4]
            ),
            onnx.helper.make_tensor_value_info(
                "mean_no_keepdims", TensorProto.FLOAT, [3]
            ),
            onnx.helper.make_tensor_value_info(
                "max_keepdims", TensorProto.FLOAT, [2, 1, 4]
            ),
            onnx.helper.make_tensor_value_info(
                "sum_noop", TensorProto.FLOAT, [2, 3, 4]
            ),
        ],
    )

    return onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", OPSET_VERSION)],
        graph=graph,
        producer_name="ONNX_Generator",
    )


if __name__ == "__main__":
    np.random.seed(42)
    np.set_printoptions(precision=8)

    onnx_model = build_model()
    file_name = "reduce_runtime_axes.onnx"

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    # -2 exercises the negative-axis path; on a rank-3 input it names dimension 1, the
    # one dimension `axes2` keeps.
    axes = np.array([-2], dtype=np.int64)
    axes2 = np.array([0, 2], dtype=np.int64)
    empty_axes = np.array([], dtype=np.int64)

    session = ReferenceEvaluator(onnx_model)
    outputs = session.run(
        None,
        {"data": data, "axes": axes, "axes2": axes2, "empty_axes": empty_axes},
    )

    print(f"Test input data: {repr(data)}")
    print(f"axes: {repr(axes)}  axes2: {repr(axes2)}  empty_axes: {repr(empty_axes)}")
    for name, value in zip(
        ["sum_keepdims", "mean_no_keepdims", "max_keepdims", "sum_noop"], outputs
    ):
        print(f"{name} {value.shape}: {repr(value)}")
