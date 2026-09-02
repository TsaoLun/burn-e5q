#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/stft/stft_with_window.onnx
#
# STFT with a Hann window applied to each frame.
# Input: [1, 32, 1], frame_step=8, frame_length=16, onesided=1.
# Window: length-16 Hann window baked as initializer.
# Output: [1, 3, 9, 2]

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    X = helper.make_tensor_value_info("signal", TensorProto.FLOAT, [1, 32, 1])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 9, 2])

    frame_step = numpy_helper.from_array(np.array(8, dtype=np.int64), name="frame_step")
    frame_length = numpy_helper.from_array(
        np.array(16, dtype=np.int64), name="frame_length"
    )
    # Periodic Hann window of length 16 (matching ONNX HannWindow periodic=1 semantics).
    n = 16
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)
    window_init = numpy_helper.from_array(hann, name="window")

    stft_node = helper.make_node(
        "STFT",
        inputs=["signal", "frame_step", "window", "frame_length"],
        outputs=["output"],
        name="stft_node",
        onesided=1,
    )

    graph = helper.make_graph(
        [stft_node],
        "stft_with_window_model",
        [X],
        [Y],
        initializer=[frame_step, frame_length, window_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, "stft_with_window.onnx")
    print("Finished exporting model to stft_with_window.onnx")

    np.random.seed(42)
    test_input = np.random.randn(1, 32, 1).astype(np.float32)
    print(f"Test input shape: {test_input.shape}")

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator("stft_with_window.onnx")
    result = session.run(None, {"signal": test_input})
    print(f"Test output shape: {result[0].shape}")
    print(f"Test output: {result[0].tolist()}")


if __name__ == "__main__":
    main()
