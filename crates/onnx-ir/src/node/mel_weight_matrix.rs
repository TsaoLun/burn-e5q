//! # MelWeightMatrix
//!
//! Generates a Mel filterbank weight matrix that re-weights a linear-frequency spectrogram
//! into a mel-scale spectrogram.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__MelWeightMatrix.html>
//!
//! ## Opset Versions
//! - **Opset 17**: Initial version.
//!
//! Mel scale definition per ONNX spec: `mel(f) = 2595 * log10(1 + f/700)`. Each triangle has a
//! peak value of 1.0. Output shape: `[floor(dft_length/2) + 1, num_mel_bins]`.
//!
//! ## Supported Configurations
//!
//! All 5 inputs may be compile-time constants OR runtime scalar inputs. When runtime, the mel
//! matrix is computed inside the generated `forward()` on each call; the work is O(n_bins *
//! num_mel_bins) which is negligible compared to the surrounding audio pipeline.

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode, TensorDataExt, TensorType};
use crate::processor::{
    ArgPreference, InputPreferences, InputSpec, NodeProcessor, NodeSpec, OutputPreferences,
    OutputSpec, ProcessError, validate_opset,
};
use crate::proto_conversion::element_type_from_proto;

const OP_NAME: &str = "MelWeightMatrix";

/// Configuration for the MelWeightMatrix operation.
#[derive(Debug, Clone)]
pub struct MelWeightMatrixConfig {
    /// Output element type (default F32).
    pub output_dtype: DType,
}

impl Default for MelWeightMatrixConfig {
    fn default() -> Self {
        Self {
            output_dtype: DType::F32,
        }
    }
}

/// Node representation for MelWeightMatrix.
#[derive(Debug, Clone, NodeBuilder)]
pub struct MelWeightMatrixNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: MelWeightMatrixConfig,
}

pub(crate) struct MelWeightMatrixProcessor;

impl MelWeightMatrixProcessor {
    fn resolve_output_dtype(node: &RawNode) -> Result<DType, ProcessError> {
        let dtype = match node.attrs.get("output_datatype") {
            Some(val) => {
                let dt_i32 = val.clone().into_i32();
                element_type_from_proto(dt_i32).map_err(|e| ProcessError::InvalidAttribute {
                    name: "output_datatype".to_string(),
                    reason: format!("{OP_NAME}: {e}"),
                })?
            }
            None => DType::F32,
        };

        // The spec allows any numeric type for output, but a mel weight matrix with integer
        // entries is almost always a bug (triangle values are in [0, 1]). We follow the same
        // policy as the window ops and accept only float types.
        if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
            return Err(ProcessError::InvalidAttribute {
                name: "output_datatype".to_string(),
                reason: format!("{OP_NAME}: must be a float type, got {dtype:?}"),
            });
        }

        Ok(dtype)
    }

    /// Validate that `arg` is a rank-0 or rank-1[1] scalar with a dtype in `allowed`.
    fn validate_scalar(
        arg: &Argument,
        name: &str,
        allowed: &[DType],
        dtype_label: &str,
    ) -> Result<(), ProcessError> {
        let is_scalar_shape = arg.ty.is_scalar()
            || matches!(&arg.ty, ArgType::Tensor(t) if t.rank == 0)
            || matches!(&arg.ty, ArgType::Tensor(t) if t.rank == 1
                && t.static_shape.as_ref().is_some_and(|s| s == &[Some(1)]));
        if !is_scalar_shape {
            return Err(ProcessError::TypeMismatch {
                expected: format!("scalar {name}"),
                actual: format!("{:?}", arg.ty),
            });
        }
        let dtype = arg.ty.elem_type();
        if !allowed.contains(&dtype) {
            return Err(ProcessError::TypeMismatch {
                expected: format!("{dtype_label} {name}"),
                actual: format!("{dtype:?}"),
            });
        }
        Ok(())
    }

    fn validate_int_scalar(arg: &Argument, name: &str) -> Result<(), ProcessError> {
        Self::validate_scalar(arg, name, &[DType::I32, DType::I64], "int32/int64")
    }

    fn validate_float_scalar(arg: &Argument, name: &str) -> Result<(), ProcessError> {
        // The ONNX spec (T2) allows F16/BF16/F32/F64 for the edge_hertz inputs, but our
        // codegen computes the mel/Hz chain in f32. Silently narrowing F64 edges to f32
        // would be a precision-loss bug that is hard to spot, and F16/BF16 do not have
        // stable native Rust scalar types for the ScalarNative preference path. We accept
        // F32 only for now; the other types can be added if a real model needs them
        // (would require a separate f64 computation path).
        Self::validate_scalar(arg, name, &[DType::F32], "float32")
    }
}

impl NodeProcessor for MelWeightMatrixProcessor {
    type Config = MelWeightMatrixConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 17,
            max_opset: None,
            inputs: InputSpec::Exact(5),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn input_preferences(
        &self,
        node: &RawNode,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        // All 5 inputs are logical scalars; codegen uses them as Rust native values.
        let mut prefs = InputPreferences::new();
        for input in node.inputs.iter().take(5) {
            prefs = prefs.add(&input.name, ArgPreference::ScalarNative);
        }
        Ok(Some(prefs))
    }

    // Note: we deliberately do not implement lift_constants. Unlike DFT/STFT, all 5 MWM inputs
    // flow through to the generated forward() body and are referenced by name. Lifting them to
    // Static would clear their names and break codegen; instead, the ScalarNative preference
    // hint converts on-device scalars to native Rust values at the pipeline boundary, which is
    // what the codegen expects.

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        validate_opset(opset, 17)?;

        Self::validate_int_scalar(&node.inputs[0], "num_mel_bins")?;
        Self::validate_int_scalar(&node.inputs[1], "dft_length")?;
        Self::validate_int_scalar(&node.inputs[2], "sample_rate")?;
        Self::validate_float_scalar(&node.inputs[3], "lower_edge_hertz")?;
        Self::validate_float_scalar(&node.inputs[4], "upper_edge_hertz")?;

        let output_dtype = Self::resolve_output_dtype(node)?;

        // When both edges are constants, reject misordered ranges up front. A negative or zero
        // span would silently collapse every triangle to the empty matrix at runtime; catching
        // it at IR time gives the user a clear error at model-load.
        if let (Some(lower_data), Some(upper_data)) =
            (node.inputs[3].value(), node.inputs[4].value())
            && let (Ok(lower), Ok(upper)) = (lower_data.scalar_f64(), upper_data.scalar_f64())
            && lower >= upper
        {
            return Err(ProcessError::Custom(format!(
                "{OP_NAME}: lower_edge_hertz ({lower}) must be strictly less than \
                 upper_edge_hertz ({upper})"
            )));
        }

        // Static shape is only knowable when dft_length and num_mel_bins are constants.
        let num_mel_bins = node.inputs[0]
            .value()
            .and_then(|d| d.scalar_i64().ok())
            .filter(|&v| v >= 0)
            .map(|v| v as usize);
        let dft_length = node.inputs[1]
            .value()
            .and_then(|d| d.scalar_i64().ok())
            .filter(|&v| v >= 0)
            .map(|v| v as usize);
        let static_shape = match (dft_length, num_mel_bins) {
            (Some(dft), Some(mel)) => Some(vec![Some(dft / 2 + 1), Some(mel)]),
            _ => None,
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: output_dtype,
            rank: 2,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let output_dtype = Self::resolve_output_dtype(node)?;
        Ok(MelWeightMatrixConfig { output_dtype })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self.extract_config(&builder, opset).unwrap_or_else(|e| {
            panic!(
                "{OP_NAME} ({}): config extraction failed: {e}",
                builder.name
            )
        });

        Node::MelWeightMatrix(MelWeightMatrixNode {
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
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;
    use crate::processor::OutputPreferences;

    fn runtime_builder() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::MelWeightMatrix, "test_mwm")
            .input_tensor_i64("num_mel_bins", 0, None)
            .input_tensor_i64("dft_length", 0, None)
            .input_tensor_i64("sample_rate", 0, None)
            .input_tensor_f32("lower_edge_hertz", 0, None)
            .input_tensor_f32("upper_edge_hertz", 0, None)
            .output_tensor_f32("output", 0, None)
    }

    #[test]
    fn test_mwm_runtime_inputs() {
        let mut node = runtime_builder().build();
        let prefs = OutputPreferences::new();
        MelWeightMatrixProcessor
            .infer_types(&mut node, 17, &prefs)
            .unwrap();
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.dtype, DType::F32);
                assert_eq!(t.static_shape, None);
            }
            _ => panic!("expected Tensor output"),
        }
    }

    #[test]
    fn test_mwm_constant_inputs_static_shape() {
        let mut node = TestNodeBuilder::new(NodeType::MelWeightMatrix, "test_mwm")
            .input_tensor_i64_data("num_mel_bins", vec![8], vec![])
            .input_tensor_i64_data("dft_length", vec![16], vec![])
            .input_tensor_i64_data("sample_rate", vec![16000], vec![])
            .input_tensor_f32_data("lower_edge_hertz", vec![0.0], vec![])
            .input_tensor_f32_data("upper_edge_hertz", vec![8000.0], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);
        let prefs = OutputPreferences::new();
        MelWeightMatrixProcessor
            .infer_types(&mut node, 17, &prefs)
            .unwrap();
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                // floor(16/2)+1 = 9, num_mel_bins = 8
                assert_eq!(t.static_shape, Some(vec![Some(9), Some(8)]));
            }
            _ => panic!("expected Tensor output"),
        }
    }

    #[test]
    fn test_mwm_output_dtype_f64() {
        let mut node = runtime_builder()
            .attr_int("output_datatype", 11) // DOUBLE
            .build();
        let prefs = OutputPreferences::new();
        MelWeightMatrixProcessor
            .infer_types(&mut node, 17, &prefs)
            .unwrap();
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.dtype, DType::F64),
            _ => panic!("expected Tensor output"),
        }
    }

    #[test]
    fn test_mwm_rejects_integer_output_dtype() {
        let mut node = runtime_builder()
            .attr_int("output_datatype", 7) // INT64
            .build();
        let prefs = OutputPreferences::new();
        let err = MelWeightMatrixProcessor
            .infer_types(&mut node, 17, &prefs)
            .unwrap_err();
        assert!(matches!(err, ProcessError::InvalidAttribute { .. }));
    }

    #[test]
    fn test_mwm_rejects_float_num_mel_bins() {
        let mut node = TestNodeBuilder::new(NodeType::MelWeightMatrix, "test_mwm")
            .input_tensor_f32("num_mel_bins", 0, None)
            .input_tensor_i64("dft_length", 0, None)
            .input_tensor_i64("sample_rate", 0, None)
            .input_tensor_f32("lower_edge_hertz", 0, None)
            .input_tensor_f32("upper_edge_hertz", 0, None)
            .output_tensor_f32("output", 0, None)
            .build();
        let prefs = OutputPreferences::new();
        let err = MelWeightMatrixProcessor
            .infer_types(&mut node, 17, &prefs)
            .unwrap_err();
        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn test_mwm_rejects_integer_lower_edge_hertz() {
        let mut node = TestNodeBuilder::new(NodeType::MelWeightMatrix, "test_mwm")
            .input_tensor_i64("num_mel_bins", 0, None)
            .input_tensor_i64("dft_length", 0, None)
            .input_tensor_i64("sample_rate", 0, None)
            .input_tensor_i64("lower_edge_hertz", 0, None)
            .input_tensor_f32("upper_edge_hertz", 0, None)
            .output_tensor_f32("output", 0, None)
            .build();
        let prefs = OutputPreferences::new();
        let err = MelWeightMatrixProcessor
            .infer_types(&mut node, 17, &prefs)
            .unwrap_err();
        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn test_mwm_rejects_f64_edge_hertz() {
        // F64 edge inputs would silently narrow to f32 in codegen. Reject at IR time.
        let mut node = TestNodeBuilder::new(NodeType::MelWeightMatrix, "test_mwm")
            .input_tensor_i64("num_mel_bins", 0, None)
            .input_tensor_i64("dft_length", 0, None)
            .input_tensor_i64("sample_rate", 0, None)
            .input_tensor_f64("lower_edge_hertz", 0, None)
            .input_tensor_f32("upper_edge_hertz", 0, None)
            .output_tensor_f32("output", 0, None)
            .build();
        let prefs = OutputPreferences::new();
        let err = MelWeightMatrixProcessor
            .infer_types(&mut node, 17, &prefs)
            .unwrap_err();
        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn test_mwm_opset_too_low() {
        let mut node = runtime_builder().build();
        let prefs = OutputPreferences::new();
        let err = MelWeightMatrixProcessor
            .infer_types(&mut node, 16, &prefs)
            .unwrap_err();
        assert!(err.to_string().contains("opset"));
    }

    #[test]
    fn test_mwm_rejects_lower_ge_upper_edge() {
        // Both edges constant and lower >= upper: IR-time rejection.
        let mut node = TestNodeBuilder::new(NodeType::MelWeightMatrix, "test_mwm")
            .input_tensor_i64_data("num_mel_bins", vec![8], vec![])
            .input_tensor_i64_data("dft_length", vec![16], vec![])
            .input_tensor_i64_data("sample_rate", vec![16000], vec![])
            .input_tensor_f32_data("lower_edge_hertz", vec![4000.0], vec![])
            .input_tensor_f32_data("upper_edge_hertz", vec![100.0], vec![])
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);
        let prefs = OutputPreferences::new();
        let err = MelWeightMatrixProcessor
            .infer_types(&mut node, 17, &prefs)
            .unwrap_err();
        assert!(
            err.to_string().contains("strictly less than"),
            "expected lower<upper error, got: {err}"
        );
    }

    #[test]
    fn test_mwm_accepts_i32_int_inputs() {
        let mut node = TestNodeBuilder::new(NodeType::MelWeightMatrix, "test_mwm")
            .input_tensor_i32("num_mel_bins", 0, None)
            .input_tensor_i32("dft_length", 0, None)
            .input_tensor_i32("sample_rate", 0, None)
            .input_tensor_f32("lower_edge_hertz", 0, None)
            .input_tensor_f32("upper_edge_hertz", 0, None)
            .output_tensor_f32("output", 0, None)
            .build();
        let prefs = OutputPreferences::new();
        MelWeightMatrixProcessor
            .infer_types(&mut node, 17, &prefs)
            .unwrap();
    }

    #[test]
    fn test_mwm_accepts_rank1_singleton_scalar() {
        // ONNX models sometimes wrap scalars as rank-1 [1]; the validator allows this shape.
        let mut node = TestNodeBuilder::new(NodeType::MelWeightMatrix, "test_mwm")
            .input_tensor_i64("num_mel_bins", 1, Some(vec![1]))
            .input_tensor_i64("dft_length", 1, Some(vec![1]))
            .input_tensor_i64("sample_rate", 1, Some(vec![1]))
            .input_tensor_f32("lower_edge_hertz", 1, Some(vec![1]))
            .input_tensor_f32("upper_edge_hertz", 1, Some(vec![1]))
            .output_tensor_f32("output", 0, None)
            .build();
        let prefs = OutputPreferences::new();
        MelWeightMatrixProcessor
            .infer_types(&mut node, 17, &prefs)
            .unwrap();
    }

    #[test]
    fn test_mwm_partial_constant_shape_is_dynamic() {
        // Only num_mel_bins is constant; dft_length is runtime -> shape must stay dynamic.
        let mut node = TestNodeBuilder::new(NodeType::MelWeightMatrix, "test_mwm")
            .input_tensor_i64_data("num_mel_bins", vec![8], vec![])
            .input_tensor_i64("dft_length", 0, None)
            .input_tensor_i64("sample_rate", 0, None)
            .input_tensor_f32("lower_edge_hertz", 0, None)
            .input_tensor_f32("upper_edge_hertz", 0, None)
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(17);
        let prefs = OutputPreferences::new();
        MelWeightMatrixProcessor
            .infer_types(&mut node, 17, &prefs)
            .unwrap();
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.static_shape, None);
            }
            _ => panic!("expected Tensor output"),
        }
    }
}
