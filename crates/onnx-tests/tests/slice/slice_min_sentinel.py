#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/slice/slice_min_sentinel.onnx
#
# Reverse slicing with the ONNX INT64_MIN sentinel for `ends`, which means
# "past the first element". Exporters emit this for `x[::-1]`.

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
    # axis 0: full reverse  -> starts=4, ends=INT64_MIN, step=-1
    # axis 1: partial reverse -> starts=2, ends=0,       step=-1
    nodes = [
        const("starts", [4, 2]),
        const("ends", [INT64_MIN, 0]),
        const("axes", [0, 1]),
        const("steps", [-1, -1]),
        helper.make_node(
            "Slice",
            name="slice_node",
            inputs=["input_tensor", "starts", "ends", "axes", "steps"],
            outputs=["output"],
        ),
    ]

    graph_def = helper.make_graph(
        nodes=nodes,
        name="SliceMinSentinelGraph",
        inputs=[
            helper.make_tensor_value_info("input_tensor", TensorProto.FLOAT, [5, 3]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [5, 2]),
        ],
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="slice_min_sentinel",
        opset_imports=[helper.make_opsetid("", 16)],
    )
    onnx.checker.check_model(model_def)
    onnx.save(model_def, "slice_min_sentinel.onnx")

    test_input = np.arange(15, dtype=np.float32).reshape(5, 3)
    result = ReferenceEvaluator(model_def).run(None, {"input_tensor": test_input})[0]
    print(f"Test input:\n{test_input}")
    print(f"Expected output:\n{result}")


if __name__ == "__main__":
    main()
