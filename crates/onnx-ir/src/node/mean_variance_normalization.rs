//! # MeanVarianceNormalization
//!
//! Performs mean-variance normalization on the input tensor using the formula
//! `Y = (X - E[X]) / sqrt(E[(X - E[X])^2])`, where the mean and variance are
//! reduced along the axes specified by the `axes` attribute (default `[0, 2, 3]`
//! i.e. normalize across batch and spatial dims, per-channel).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__MeanVarianceNormalization.html>
//!
//! ## Type Constraints
//! - T: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
//!
//! ## Opset Versions
//! - **Opset 9**: Initial version (types: float16, float, double)
//! - **Opset 13**: Added bfloat16 type support
use crate::ir::{ArgType, Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use burn_tensor::DType;
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for MeanVarianceNormalization.
#[derive(Debug, Clone, new)]
pub struct MeanVarianceNormalizationConfig {
    /// Axes along which mean and variance are computed. Resolved to positive
    /// indices, sorted ascending, and deduplicated. Defaults to `[0, 2, 3]`
    /// per the ONNX spec.
    pub axes: Vec<usize>,
}

/// Node representation for the MeanVarianceNormalization operation.
#[derive(Debug, Clone, NodeBuilder)]
pub struct MeanVarianceNormalizationNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: MeanVarianceNormalizationConfig,
}

pub(crate) struct MeanVarianceNormalizationProcessor;

impl NodeProcessor for MeanVarianceNormalizationProcessor {
    type Config = MeanVarianceNormalizationConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 9,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let arg = node
            .inputs
            .first()
            .ok_or_else(|| ProcessError::MissingInput("Missing input X".to_string()))?;
        let ArgType::Tensor(ref tensor_ty) = arg.ty else {
            return Err(ProcessError::TypeMismatch {
                expected: "Input should be a tensor".to_string(),
                actual: format!("{:?}", arg.ty),
            });
        };

        let allowed = if opset >= 13 {
            matches!(
                tensor_ty.dtype,
                DType::BF16 | DType::F16 | DType::F32 | DType::F64
            )
        } else {
            matches!(tensor_ty.dtype, DType::F16 | DType::F32 | DType::F64)
        };
        if !allowed {
            return Err(ProcessError::TypeMismatch {
                expected: "Floating-point tensor dtype".to_string(),
                actual: format!("{:?}", tensor_ty.dtype),
            });
        }

        // Run axes validation for its side effect of surfacing out-of-range
        // errors during type inference. The resolved axes are rebuilt later in
        // `extract_config`; we discard the value here rather than caching it.
        let rank = tensor_ty.rank;
        extract_axes(node, rank)?;

        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let rank = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.rank,
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", other),
                });
            }
        };

        let axes = extract_axes(node, rank)?;
        Ok(MeanVarianceNormalizationConfig::new(axes))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::MeanVarianceNormalization(MeanVarianceNormalizationNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

/// Parse the `axes` attribute, resolve negative indices against `rank`, sort, and
/// deduplicate. Falls back to `[0, 2, 3]` per the ONNX default.
fn extract_axes(node: &RawNode, rank: usize) -> Result<Vec<usize>, ProcessError> {
    let (raw_axes, from_default): (Vec<i64>, bool) = match node.attrs.get("axes") {
        Some(value) => (value.clone().into_i64s(), false),
        None => (vec![0, 2, 3], true),
    };

    let rank_i64 = rank as i64;
    let mut axes: Vec<usize> = Vec::with_capacity(raw_axes.len());
    for axis in raw_axes {
        let resolved = if axis < 0 { axis + rank_i64 } else { axis };
        if resolved < 0 || resolved >= rank_i64 {
            let hint = if from_default {
                " (default axes [0, 2, 3] assume a rank-4 NCHW input; specify `axes` explicitly for other ranks)"
            } else {
                ""
            };
            return Err(ProcessError::InvalidAttribute {
                name: "axes".to_string(),
                reason: format!("axis {axis} is out of range for tensor of rank {rank}{hint}"),
            });
        }
        axes.push(resolved as usize);
    }

    axes.sort();
    axes.dedup();

    Ok(axes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn build_node(rank: usize, axes: Option<Vec<i64>>) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::MeanVarianceNormalization, "mvn")
            .input_tensor_f32("X", rank, None)
            .output_tensor_f32("Y", rank, None);
        if let Some(axes) = axes {
            builder = builder.attr_ints("axes", axes);
        }
        builder.build()
    }

    #[test]
    fn default_axes() {
        let node = build_node(4, None);
        let config = MeanVarianceNormalizationProcessor
            .extract_config(&node, 13)
            .unwrap();
        assert_eq!(config.axes, vec![0, 2, 3]);
    }

    #[test]
    fn custom_axes_sorted_and_dedup() {
        let node = build_node(4, Some(vec![3, 0, 2, 0]));
        let config = MeanVarianceNormalizationProcessor
            .extract_config(&node, 13)
            .unwrap();
        assert_eq!(config.axes, vec![0, 2, 3]);
    }

    #[test]
    fn negative_axes_resolved() {
        let node = build_node(4, Some(vec![-4, -2, -1]));
        let config = MeanVarianceNormalizationProcessor
            .extract_config(&node, 13)
            .unwrap();
        assert_eq!(config.axes, vec![0, 2, 3]);
    }

    #[test]
    fn out_of_range_axis_errors() {
        let node = build_node(4, Some(vec![4]));
        let err = MeanVarianceNormalizationProcessor
            .extract_config(&node, 13)
            .unwrap_err();
        assert!(matches!(
            err,
            ProcessError::InvalidAttribute { ref name, .. } if name == "axes"
        ));
    }

    #[test]
    fn infer_preserves_shape_and_dtype() {
        let mut node = TestNodeBuilder::new(NodeType::MeanVarianceNormalization, "mvn")
            .input_tensor_f32("X", 4, Some(vec![2, 3, 5, 7]))
            .output_tensor_f32("Y", 4, None)
            .build();
        let prefs = OutputPreferences::new();
        MeanVarianceNormalizationProcessor
            .infer_types(&mut node, 13, &prefs)
            .unwrap();
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::F32);
                assert_eq!(t.rank, 4);
                assert_eq!(
                    t.static_shape,
                    Some(vec![Some(2), Some(3), Some(5), Some(7)])
                );
            }
            _ => panic!("expected tensor output"),
        }
    }

    #[test]
    fn rejects_non_float_dtype() {
        let mut node = TestNodeBuilder::new(NodeType::MeanVarianceNormalization, "mvn")
            .input_tensor_i32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .build();
        let prefs = OutputPreferences::new();
        let err = MeanVarianceNormalizationProcessor
            .infer_types(&mut node, 13, &prefs)
            .unwrap_err();
        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn rejects_bfloat16_before_opset_13() {
        let mut node = TestNodeBuilder::new(NodeType::MeanVarianceNormalization, "mvn")
            .input_tensor_bf16("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .build();
        let prefs = OutputPreferences::new();
        let err = MeanVarianceNormalizationProcessor
            .infer_types(&mut node, 9, &prefs)
            .unwrap_err();
        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }
}
