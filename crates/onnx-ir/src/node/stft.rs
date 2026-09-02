//! # STFT (Short-Time Fourier Transform)
//!
//! Computes the Short-time Fourier Transform of the signal.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__STFT.html>
//!
//! ## Opset Versions
//! - **Opset 17**: Initial version.
//!
//! ## Supported Configurations
//!
//! Only real-input STFT maps to Burn's `stft` API:
//! - Real input: rank-3 `[batch, signal_length, 1]` (ONNX spec form) or rank-2
//!   `[batch, signal_length]` (produced by PyTorch's ONNX exporter and accepted
//!   by ONNX Runtime), `onesided = 0 | 1`.
//! - `frame_step` and `frame_length` must be compile-time constants (burn-onnx
//!   codegen needs them as `usize` values).
//! - Optional window tensor, rank 1, length equal to `frame_length`.
//!
//! Power-of-two `frame_length` takes Burn's fast `burn::tensor::signal::stft`
//! path (O(N log N)). Non-power-of-two `frame_length` uses a matrix-based DFT
//! emitted directly in burn-onnx codegen (O(N²) per frame, but N is typically
//! small - e.g. 20 in Kokoro). Burn's upstream `stft` cannot be used for
//! non-power-of-two `n_fft` until Bluestein's algorithm lands
//! ([tracel-ai/burn#4865](https://github.com/tracel-ai/burn/issues/4865)).
//!
//! Not supported (rejected with a clear error):
//! - Complex-to-complex STFT (signal rank 3, trailing dim = 2): Burn's STFT
//!   has no complex input path.
//! - Runtime `frame_step` / `frame_length`.
//!
//! ## `frame_length` default
//!
//! The ONNX spec's default for `frame_length` when absent is `signal_length`.
//! That would require a runtime-sized DFT, so we instead fall back to the
//! `window` tensor's static shape when `frame_length` is absent. Real ONNX
//! models that omit `frame_length` virtually always provide a window, so this
//! deviation is safe in practice.

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorData, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, validate_opset,
};

const OP_NAME: &str = "STFT";

/// Extract a scalar integer from tensor data, supporting both i32 and i64 dtypes.
fn extract_scalar_int(data: TensorData, name: &str) -> Result<i64, ProcessError> {
    if let Ok(slice) = data.as_slice::<i64>() {
        slice.first().copied().ok_or_else(|| {
            ProcessError::Custom(format!(
                "{OP_NAME}: {name} constant must contain at least one element"
            ))
        })
    } else if let Ok(slice) = data.as_slice::<i32>() {
        slice.first().copied().map(i64::from).ok_or_else(|| {
            ProcessError::Custom(format!(
                "{OP_NAME}: {name} constant must contain at least one element"
            ))
        })
    } else {
        Err(ProcessError::Custom(format!(
            "{OP_NAME}: {name} constant must have type int32 or int64"
        )))
    }
}

/// Configuration for the STFT operation.
#[derive(Debug, Clone, Default)]
pub struct StftConfig {
    /// If true, produces onesided output (`n_fft / 2 + 1` frequency bins).
    pub onesided: bool,
    /// Number of samples to step between successive DFTs (Burn's `hop_length`).
    pub frame_step: usize,
    /// Size of each DFT window (Burn's `n_fft`). Must be a power of two.
    pub frame_length: usize,
    /// Whether a window tensor is provided as input.
    pub has_window: bool,
}

/// Node representation for the STFT operation.
#[derive(Debug, Clone, NodeBuilder)]
pub struct StftNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: StftConfig,
}

pub(crate) struct StftProcessor;

impl StftProcessor {
    /// Extract `frame_step` from input[1] (required, must be a positive scalar constant).
    fn extract_frame_step(node: &RawNode) -> Result<usize, ProcessError> {
        let input = node.inputs.get(1).ok_or_else(|| {
            ProcessError::Custom(format!("{OP_NAME}: missing required frame_step input"))
        })?;
        let data = input.value().ok_or_else(|| {
            ProcessError::Custom(format!(
                "{OP_NAME}: frame_step must be a compile-time constant"
            ))
        })?;
        let val = extract_scalar_int(data, "frame_step")?;
        if val <= 0 {
            return Err(ProcessError::Custom(format!(
                "{OP_NAME}: frame_step must be a positive integer, got {val}"
            )));
        }
        usize::try_from(val).map_err(|_| {
            ProcessError::Custom(format!(
                "{OP_NAME}: frame_step must fit in usize, got {val}"
            ))
        })
    }

    /// Extract `frame_length` from input[3] if present, otherwise infer from `window` shape.
    ///
    /// Returns `None` only if neither is provided (which the caller rejects).
    fn extract_frame_length(node: &RawNode) -> Result<Option<usize>, ProcessError> {
        if let Some(input) = node.inputs.get(3)
            && !input.is_optional()
        {
            let data = input.value().ok_or_else(|| {
                ProcessError::Custom(format!(
                    "{OP_NAME}: frame_length must be a compile-time constant"
                ))
            })?;
            let val = extract_scalar_int(data, "frame_length")?;
            if val <= 0 {
                return Err(ProcessError::Custom(format!(
                    "{OP_NAME}: frame_length must be a positive integer, got {val}"
                )));
            }
            let val = usize::try_from(val).map_err(|_| {
                ProcessError::Custom(format!(
                    "{OP_NAME}: frame_length must fit in usize, got {val}"
                ))
            })?;
            return Ok(Some(val));
        }

        // Fall back to inferring from the window's static shape.
        if let Some(window) = node.inputs.get(2)
            && !window.is_optional()
            && let ArgType::Tensor(t) = &window.ty
            && let Some(shape) = &t.static_shape
            && let Some(Some(w)) = shape.first()
        {
            return Ok(Some(*w));
        }

        Ok(None)
    }

    /// Compute the output shape `[batch, n_frames, n_freqs, 2]` when the signal length is known.
    fn compute_output_shape(
        signal_shape: &[Option<usize>],
        frame_step: usize,
        frame_length: usize,
        onesided: bool,
    ) -> Vec<Option<usize>> {
        let batch = signal_shape.first().copied().flatten();
        let signal_len = signal_shape.get(1).copied().flatten();

        let n_frames = signal_len.map(|l| {
            if l < frame_length {
                0
            } else {
                1 + (l - frame_length) / frame_step
            }
        });

        let n_freqs = if onesided {
            frame_length / 2 + 1
        } else {
            frame_length
        };

        vec![batch, n_frames, Some(n_freqs), Some(2)]
    }
}

impl NodeProcessor for StftProcessor {
    type Config = StftConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 17,
            max_opset: None,
            inputs: InputSpec::Range(2, 4),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        if let Some(input) = node.inputs.get(1)
            && !input.is_optional()
            && input.is_constant()
        {
            node.inputs[1].to_static()?;
        }

        if let Some(input) = node.inputs.get(3)
            && !input.is_optional()
            && input.is_constant()
        {
            node.inputs[3].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        validate_opset(opset, 17)?;

        let signal_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.clone(),
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        };

        // Accept rank 3 (ONNX spec: [batch, signal_length, 1|2]) and rank 2
        // ([batch, signal_length]). PyTorch's ONNX exporter emits the rank-2
        // form; ORT accepts it as real-valued.
        match signal_tensor.rank {
            2 => { /* implicit real input, no trailing dim to inspect */ }
            3 => {
                let is_real_input = match &signal_tensor.static_shape {
                    Some(shape) => match shape.last() {
                        Some(Some(1)) => true,
                        Some(Some(2)) => false,
                        Some(Some(d)) => {
                            return Err(ProcessError::Custom(format!(
                                "{OP_NAME}: signal last dimension must be 1 (real) or 2 (complex), got {d}"
                            )));
                        }
                        _ => {
                            return Err(ProcessError::Custom(format!(
                                "{OP_NAME}: signal last dimension must be statically known as 1 or 2"
                            )));
                        }
                    },
                    None => {
                        return Err(ProcessError::Custom(format!(
                            "{OP_NAME}: signal shape must be statically known \
                             (last dim determines real/complex)"
                        )));
                    }
                };

                if !is_real_input {
                    return Err(ProcessError::Custom(format!(
                        "{OP_NAME}: complex-to-complex STFT is not supported. \
                         Burn's stft requires real-valued input."
                    )));
                }
            }
            r => {
                return Err(ProcessError::Custom(format!(
                    "{OP_NAME}: signal input must have rank 2 [batch, signal_length] \
                     or rank 3 [batch, signal_length, 1|2], got rank {r}"
                )));
            }
        }

        // Optional window (input[2]) must be rank 1 if present.
        if let Some(window) = node.inputs.get(2)
            && !window.is_optional()
        {
            match &window.ty {
                ArgType::Tensor(t) => {
                    if t.rank != 1 {
                        return Err(ProcessError::Custom(format!(
                            "{OP_NAME}: window must have rank 1, got rank {}",
                            t.rank
                        )));
                    }
                }
                other => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Tensor for window".to_string(),
                        actual: format!("{other:?}"),
                    });
                }
            }
        }

        let frame_step = Self::extract_frame_step(node)?;
        let frame_length = Self::extract_frame_length(node)?.ok_or_else(|| {
            ProcessError::Custom(format!(
                "{OP_NAME}: frame_length must be provided as a constant input \
                 or inferable from a window with a static shape"
            ))
        })?;

        // Non-power-of-two frame_length is handled in burn-onnx codegen via a
        // matrix DFT (upstream Burn's stft only supports pow2 n_fft pending
        // tracel-ai/burn#4865). Only reject `0`, which is nonsensical.
        if frame_length == 0 {
            return Err(ProcessError::Custom(format!(
                "{OP_NAME}: frame_length must be >= 1, got {frame_length}"
            )));
        }

        if frame_step > frame_length {
            return Err(ProcessError::Custom(format!(
                "{OP_NAME}: frame_step ({frame_step}) must be <= frame_length ({frame_length}) \
                 to satisfy the overlap-add constraint"
            )));
        }

        // Cross-check window length against frame_length when both are statically known.
        if let Some(window) = node.inputs.get(2)
            && !window.is_optional()
            && let ArgType::Tensor(t) = &window.ty
            && let Some(shape) = &t.static_shape
            && let Some(Some(w)) = shape.first()
            && *w != frame_length
        {
            return Err(ProcessError::Custom(format!(
                "{OP_NAME}: window length ({w}) must equal frame_length ({frame_length})"
            )));
        }

        let onesided = node
            .attrs
            .get("onesided")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(true);

        let static_shape = signal_tensor
            .static_shape
            .as_ref()
            .map(|s| Self::compute_output_shape(s, frame_step, frame_length, onesided));

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: signal_tensor.dtype,
            rank: 4,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let frame_step = Self::extract_frame_step(node)?;
        let frame_length = Self::extract_frame_length(node)?.ok_or_else(|| {
            ProcessError::Custom(format!("{OP_NAME}: frame_length could not be resolved"))
        })?;
        let onesided = node
            .attrs
            .get("onesided")
            .map(|v| v.clone().into_i64() != 0)
            .unwrap_or(true);
        let has_window = matches!(
            node.inputs.get(2),
            Some(input) if !input.is_optional()
        );

        Ok(StftConfig {
            onesided,
            frame_step,
            frame_length,
            has_window,
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self.extract_config(&builder, opset).unwrap_or_else(|e| {
            panic!(
                "{OP_NAME} ({}): config extraction failed: {e}",
                builder.name
            )
        });

        // frame_step (input[1]) and frame_length (input[3]) were lifted to Static by
        // lift_constants, so they are dropped from the generated forward() signature.
        Node::Stft(StftNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;
    use crate::processor::OutputPreferences;

    fn builder_with_signal(signal_shape: Vec<usize>) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Stft, "test_stft").input_tensor_f32(
            "signal",
            signal_shape.len(),
            Some(signal_shape),
        )
    }

    #[test]
    fn test_stft_real_onesided_default() {
        let mut node = builder_with_signal(vec![1, 64, 1])
            .input_tensor_i64_data("frame_step", vec![4], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 4);
                let shape = t.static_shape.as_ref().unwrap();
                // n_frames = 1 + (64 - 16) / 4 = 13, n_freqs onesided = 16/2+1 = 9
                assert_eq!(shape, &vec![Some(1), Some(13), Some(9), Some(2)]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_stft_real_full() {
        let mut node = builder_with_signal(vec![2, 32, 1])
            .input_tensor_i64_data("frame_step", vec![8], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![8], vec![])
            .output_tensor_f32("output", 0, None)
            .attr_int("onesided", 0)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                let shape = t.static_shape.as_ref().unwrap();
                // n_frames = 1 + (32-8)/8 = 4, n_freqs full = 8
                assert_eq!(shape, &vec![Some(2), Some(4), Some(8), Some(2)]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_stft_with_window() {
        let mut node = builder_with_signal(vec![1, 128, 1])
            .input_tensor_i64_data("frame_step", vec![16], vec![])
            .input_tensor_f32("window", 1, Some(vec![32]))
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::I64,
                    rank: 0,
                    static_shape: None,
                }),
            )
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        let config = processor.extract_config(&node, 17).unwrap();
        assert_eq!(config.frame_step, 16);
        assert_eq!(config.frame_length, 32);
        assert!(config.has_window);
    }

    #[test]
    fn test_stft_real_rank2_input() {
        // PyTorch's ONNX exporter emits real STFT signal as rank 2
        // [batch, signal_length] with no trailing 1. ORT accepts it and
        // so should we.
        let mut node = builder_with_signal(vec![1, 64])
            .input_tensor_i64_data("frame_step", vec![4], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 4);
                let shape = t.static_shape.as_ref().unwrap();
                // Same as the rank-3 real case: batch=1, n_frames=13, n_freqs=9
                assert_eq!(shape, &vec![Some(1), Some(13), Some(9), Some(2)]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_stft_rank1_rejected() {
        let mut node = builder_with_signal(vec![64])
            .input_tensor_i64_data("frame_step", vec![4], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        let err = processor.infer_types(&mut node, 17, &prefs).unwrap_err();
        assert!(err.to_string().contains("rank 1"));
    }

    #[test]
    fn test_stft_complex_rejected() {
        let mut node = builder_with_signal(vec![1, 64, 2])
            .input_tensor_i64_data("frame_step", vec![4], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("complex"));
    }

    #[test]
    fn test_stft_non_power_of_two_accepted() {
        // Non-pow2 frame_length (20, matching Kokoro) used to be rejected
        // because upstream Burn's stft is pow2-only. burn-onnx now handles
        // non-pow2 via a matmul DFT path, so onnx-ir accepts it and leaves
        // the branching to codegen.
        let mut node = builder_with_signal(vec![1, 128, 1])
            .input_tensor_i64_data("frame_step", vec![5], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![20], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 17, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                let shape = t.static_shape.as_ref().unwrap();
                // n_frames = 1 + (128 - 20) / 5 = 22, n_freqs onesided = 11
                assert_eq!(shape, &vec![Some(1), Some(22), Some(11), Some(2)]);
            }
            _ => panic!("Expected Tensor output"),
        }

        let config = processor.extract_config(&node, 17).unwrap();
        assert_eq!(config.frame_length, 20);
        assert_eq!(config.frame_step, 5);
    }

    #[test]
    fn test_stft_runtime_frame_step_rejected() {
        let mut node = builder_with_signal(vec![1, 64, 1])
            .input_tensor_i64("frame_step", 0, None) // runtime
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("compile-time constant")
        );
    }

    #[test]
    fn test_stft_frame_step_gt_frame_length_rejected() {
        let mut node = builder_with_signal(vec![1, 64, 1])
            .input_tensor_i64_data("frame_step", vec![32], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("overlap-add"));
    }

    #[test]
    fn test_stft_window_mismatch_rejected() {
        let mut node = builder_with_signal(vec![1, 128, 1])
            .input_tensor_i64_data("frame_step", vec![4], vec![])
            .input_tensor_f32("window", 1, Some(vec![20])) // mismatch
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 17, &prefs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("window length"));
    }

    #[test]
    fn test_stft_infer_frame_length_from_window() {
        let node = builder_with_signal(vec![1, 128, 1])
            .input_tensor_i64_data("frame_step", vec![8], vec![])
            .input_tensor_f32("window", 1, Some(vec![16]))
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);

        let processor = StftProcessor;
        let config = processor.extract_config(&node, 17).unwrap();
        assert_eq!(config.frame_length, 16);
        assert!(config.has_window);
    }

    #[test]
    fn test_stft_opset_too_low() {
        let mut node = builder_with_signal(vec![1, 64, 1])
            .input_tensor_i64_data("frame_step", vec![4], vec![])
            .add_input(
                "",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .input_tensor_i64_data("frame_length", vec![16], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(16);

        let processor = StftProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("opset"));
    }

    // compute_output_shape boundary tests: exercise the three n_frames branches
    // (signal < frame_length, signal == frame_length, non-exact division) that
    // the higher-level tests all skip because their lengths divide evenly.

    #[test]
    fn test_stft_shape_signal_shorter_than_frame_length() {
        // signal_length (8) < frame_length (16) -> n_frames = 0
        let shape = StftProcessor::compute_output_shape(&[Some(1), Some(8), Some(1)], 4, 16, true);
        assert_eq!(shape, vec![Some(1), Some(0), Some(9), Some(2)]);
    }

    #[test]
    fn test_stft_shape_signal_equals_frame_length() {
        // signal_length == frame_length -> exactly 1 frame
        let shape = StftProcessor::compute_output_shape(&[Some(1), Some(16), Some(1)], 8, 16, true);
        assert_eq!(shape, vec![Some(1), Some(1), Some(9), Some(2)]);
    }

    #[test]
    fn test_stft_shape_non_exact_division() {
        // signal_length=33, frame_length=16, frame_step=8:
        // (33 - 16) / 8 = 17/8 = 2 (floor) -> n_frames = 3
        let shape = StftProcessor::compute_output_shape(&[Some(1), Some(33), Some(1)], 8, 16, true);
        assert_eq!(shape, vec![Some(1), Some(3), Some(9), Some(2)]);
    }
}
