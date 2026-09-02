#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/stft/stft_basic.onnx
#
# Basic STFT: real input, onesided, no window, constant frame_step/frame_length.
# Input: [1, 32, 1] real signal
# frame_step=8, frame_length=16 -> n_frames = 1 + (32-16)/8 = 3
# onesided=1 -> n_freqs = 16/2 + 1 = 9
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

    stft_node = helper.make_node(
        "STFT",
        inputs=["signal", "frame_step", "", "frame_length"],
        outputs=["output"],
        name="stft_node",
        onesided=1,
    )

    graph = helper.make_graph(
        [stft_node],
        "stft_basic_model",
        [X],
        [Y],
        initializer=[frame_step, frame_length],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, "stft_basic.onnx")
    print("Finished exporting model to stft_basic.onnx")

    # Test with a deterministic signal for reproducible output.
    np.random.seed(42)
    test_input = np.random.randn(1, 32, 1).astype(np.float32)
    print(f"Test input shape: {test_input.shape}")

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator("stft_basic.onnx")
    result = session.run(None, {"signal": test_input})
    print(f"Test output shape: {result[0].shape}")
    print(f"Test input flat: {test_input.flatten().tolist()}")
    print(f"Test output: {result[0].tolist()}")


if __name__ == "__main__":
    main()
