#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/slice/slice_shape_reverse.onnx
#
# Reverse slicing of a Shape value rather than a tensor. This path builds a
# fixed-size array whose length comes from onnx-ir's shape inference, so a
# bounds disagreement surfaces as a panic in the generated code rather than a
# wrong value.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

INT64_MIN = -9223372036854775808


def const(name: str, vals: list[int]):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=f"{name}_constant",
        value=helper.make_tensor(
            name=name,
            data_type=TensorProto.INT64,
            dims=[len(vals)],
            vals=vals,
        ),
    )


def main() -> None:
    nodes = [helper.make_node("Shape", ["input_tensor"], ["shape"], name="shape_node")]
    # shape[::-1] -> the whole shape reversed
    nodes += [
        const("r_starts", [5]),
        const("r_ends", [INT64_MIN]),
        const("r_steps", [-1]),
        helper.make_node(
            "Slice",
            name="slice_reversed",
            inputs=["shape", "r_starts", "r_ends", "", "r_steps"],
            outputs=["reversed"],
        ),
    ]
    # shape[5::-2] -> entries 5, 3, 1
    nodes += [
        const("s_starts", [5]),
        const("s_ends", [INT64_MIN]),
        const("s_steps", [-2]),
        helper.make_node(
            "Slice",
            name="slice_strided",
            inputs=["shape", "s_starts", "s_ends", "", "s_steps"],
            outputs=["strided"],
        ),
    ]

    graph_def = helper.make_graph(
        nodes=nodes,
        name="SliceShapeReverseGraph",
        inputs=[
            helper.make_tensor_value_info(
                "input_tensor", TensorProto.FLOAT, [2, 3, 4, 5, 6, 7]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info("reversed", TensorProto.INT64, [6]),
            helper.make_tensor_value_info("strided", TensorProto.INT64, [3]),
        ],
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="slice_shape_reverse",
        opset_imports=[helper.make_opsetid("", 16)],
    )
    onnx.checker.check_model(model_def)
    onnx.save(model_def, "slice_shape_reverse.onnx")

    test_input = np.zeros((2, 3, 4, 5, 6, 7), dtype=np.float32)
    results = ReferenceEvaluator(model_def).run(None, {"input_tensor": test_input})
    print(f"reversed = {results[0]}")
    print(f"strided  = {results[1]}")


if __name__ == "__main__":
    main()
