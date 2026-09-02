#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/slice/slice_reverse_steps.onnx
#
# Reverse slicing with |step| > 1, and the reverse slice ONNX evaluates to
# nothing. Burn anchors a reverse traversal at the top of the range and walks
# down, so a range whose length is not a multiple of the step is the case that
# would expose a misaligned stride.

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


def slice_node(tag: str, starts, ends, axes, steps, out):
    return [
        const(f"{tag}_starts", starts),
        const(f"{tag}_ends", ends),
        const(f"{tag}_axes", axes),
        const(f"{tag}_steps", steps),
        helper.make_node(
            "Slice",
            name=f"slice_{tag}",
            inputs=["input_tensor", f"{tag}_starts", f"{tag}_ends", f"{tag}_axes", f"{tag}_steps"],
            outputs=[out],
        ),
    ]


def main() -> None:
    nodes = []
    # x[6::-3] over dim 8 -> rows 6, 3, 0. 7 is not a multiple of 3, so a
    # stride anchored at the bottom of the range would give 6, 3 or 7, 4, 1.
    nodes += slice_node("a", [6], [INT64_MIN], [0], [-3], "out_step3")
    # x[:, 5:1:-2] over dim 6 -> cols 5, 3.
    nodes += slice_node("b", [5], [1], [1], [-2], "out_step2")
    # x[0:8:-1] selects nothing: ONNX stops before `ends` walking backwards.
    nodes += slice_node("c", [0], [8], [0], [-1], "out_empty")

    graph_def = helper.make_graph(
        nodes=nodes,
        name="SliceReverseStepsGraph",
        inputs=[
            helper.make_tensor_value_info("input_tensor", TensorProto.FLOAT, [8, 6]),
        ],
        outputs=[
            helper.make_tensor_value_info("out_step3", TensorProto.FLOAT, [3, 6]),
            helper.make_tensor_value_info("out_step2", TensorProto.FLOAT, [8, 2]),
            helper.make_tensor_value_info("out_empty", TensorProto.FLOAT, [0, 6]),
        ],
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="slice_reverse_steps",
        opset_imports=[helper.make_opsetid("", 16)],
    )
    onnx.checker.check_model(model_def)
    onnx.save(model_def, "slice_reverse_steps.onnx")

    test_input = np.arange(48, dtype=np.float32).reshape(8, 6)
    results = ReferenceEvaluator(model_def).run(None, {"input_tensor": test_input})
    for name, r in zip(["out_step3", "out_step2", "out_empty"], results):
        print(f"{name} shape={r.shape}\n{r}")


if __name__ == "__main__":
    main()
