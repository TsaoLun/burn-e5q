#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
# ]
# ///

# used to generate model: onnx-tests/tests/blackman_window/blackman_window.onnx
#
# Generates a Blackman window of size 10 (periodic, float32 output).

import onnx
from onnx import TensorProto, helper


def main():
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [10])

    size_const = helper.make_tensor("size", TensorProto.INT64, [], [10])

    blackman_node = helper.make_node(
        "BlackmanWindow",
        inputs=["size"],
        outputs=["output"],
        name="blackman_node",
        periodic=1,
        output_datatype=TensorProto.FLOAT,
    )

    graph = helper.make_graph(
        [blackman_node],
        "blackman_window_model",
        [],
        [Y],
        initializer=[size_const],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, "blackman_window.onnx")
    print("Finished exporting model to blackman_window.onnx")

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator("blackman_window.onnx")
    result = session.run(None, {})
    print(f"Test output data shape: {result[0].shape}")
    print(f"Test output data: {result[0].tolist()}")


if __name__ == "__main__":
    main()
