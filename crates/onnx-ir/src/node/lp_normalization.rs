//! # LpNormalization
//!
//! For each 1-D slice along `axis`, divides the slice by its Lp-norm:
//! `Y = X / ||X||_p`. The norm is computed per-slice (reducing `axis` to
//! size 1) so the division broadcasts against the original shape. A slice of
//! all zeros yields NaN/Inf, matching ONNX Runtime (the ONNX reference
//! evaluator adds a `where(norm == 0, 0, ...)` guard that the spec does not
//! require).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__LpNormalization.html>
//!
//! ## Type Constraints
//! - T: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version (types: float16, float, double)
//! - **Opset 22**: Added bfloat16 type support
use crate::ir::{ArgType, Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use burn_tensor::DType;
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for LpNormalization.
#[derive(Debug, Clone, new)]
pub struct LpNormalizationConfig {
    /// Axis along which to compute the p-norm. Resolved to a positive index.
    /// Defaults to the last axis (`rank - 1`) per the ONNX spec.
    pub axis: usize,
    /// Order of the norm. ONNX restricts this to 1 (L1) or 2 (L2).
    pub p: i64,
}

/// Node representation for the LpNormalization operation.
#[derive(Debug, Clone, NodeBuilder)]
pub struct LpNormalizationNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: LpNormalizationConfig,
}

pub(crate) struct LpNormalizationProcessor;

impl NodeProcessor for LpNormalizationProcessor {
    type Config = LpNormalizationConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
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
            .ok_or_else(|| ProcessError::MissingInput("input".to_string()))?;
        let ArgType::Tensor(ref tensor_ty) = arg.ty else {
            return Err(ProcessError::TypeMismatch {
                expected: "Input should be a tensor".to_string(),
                actual: format!("{:?}", arg.ty),
            });
        };

        let allowed = if opset >= 22 {
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

        // Validate here so malformed graphs fail before codegen.
        let rank = tensor_ty.rank;
        extract_axis(node, rank)?;
        extract_p(node)?;

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

        let axis = extract_axis(node, rank)?;
        let p = extract_p(node)?;
        Ok(LpNormalizationConfig::new(axis, p))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::LpNormalization(LpNormalizationNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

/// Parse `axis`, resolving negative values against `rank`. Defaults to `-1`
/// (last axis) per the ONNX spec.
fn extract_axis(node: &RawNode, rank: usize) -> Result<usize, ProcessError> {
    let raw = node
        .attrs
        .get("axis")
        .map(|v| v.clone().into_i64())
        .unwrap_or(-1);

    let rank_i64 = rank as i64;
    let resolved = if raw < 0 { raw + rank_i64 } else { raw };
    if resolved < 0 || resolved >= rank_i64 {
        return Err(ProcessError::InvalidAttribute {
            name: "axis".to_string(),
            reason: format!("axis {raw} is out of range for tensor of rank {rank}"),
        });
    }
    Ok(resolved as usize)
}

/// Parse `p`, which ONNX restricts to 1 or 2. Defaults to 2 per the ONNX spec.
fn extract_p(node: &RawNode) -> Result<i64, ProcessError> {
    let p = node
        .attrs
        .get("p")
        .map(|v| v.clone().into_i64())
        .unwrap_or(2);

    if p != 1 && p != 2 {
        return Err(ProcessError::InvalidAttribute {
            name: "p".to_string(),
            reason: format!("only p=1 or p=2 are supported by ONNX, got {p}"),
        });
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn build_node(rank: usize, axis: Option<i64>, p: Option<i64>) -> RawNode {
        let mut builder = TestNodeBuilder::new(NodeType::LpNormalization, "lpnorm")
            .input_tensor_f32("X", rank, None)
            .output_tensor_f32("Y", rank, None);
        if let Some(axis) = axis {
            builder = builder.attr_int("axis", axis);
        }
        if let Some(p) = p {
            builder = builder.attr_int("p", p);
        }
        builder.build()
    }

    #[test]
    fn default_axis_and_p() {
        let node = build_node(3, None, None);
        let config = LpNormalizationProcessor.extract_config(&node, 1).unwrap();
        assert_eq!(config.axis, 2);
        assert_eq!(config.p, 2);
    }

    #[test]
    fn l1_norm_along_axis_0() {
        let node = build_node(4, Some(0), Some(1));
        let config = LpNormalizationProcessor.extract_config(&node, 1).unwrap();
        assert_eq!(config.axis, 0);
        assert_eq!(config.p, 1);
    }

    #[test]
    fn negative_axis_resolved() {
        let node = build_node(4, Some(-2), None);
        let config = LpNormalizationProcessor.extract_config(&node, 1).unwrap();
        assert_eq!(config.axis, 2);
    }

    #[test]
    fn negative_axis_boundary() {
        // axis=-rank resolves to 0; axis=-rank-1 is out of range.
        let node = build_node(3, Some(-3), None);
        let config = LpNormalizationProcessor.extract_config(&node, 1).unwrap();
        assert_eq!(config.axis, 0);

        let node = build_node(3, Some(-4), None);
        let err = LpNormalizationProcessor
            .extract_config(&node, 1)
            .unwrap_err();
        assert!(matches!(
            err,
            ProcessError::InvalidAttribute { ref name, .. } if name == "axis"
        ));
    }

    #[test]
    fn out_of_range_axis_errors() {
        let node = build_node(3, Some(5), None);
        let err = LpNormalizationProcessor
            .extract_config(&node, 1)
            .unwrap_err();
        assert!(matches!(
            err,
            ProcessError::InvalidAttribute { ref name, .. } if name == "axis"
        ));
    }

    #[test]
    fn invalid_p_errors() {
        let node = build_node(3, None, Some(3));
        let err = LpNormalizationProcessor
            .extract_config(&node, 1)
            .unwrap_err();
        assert!(matches!(
            err,
            ProcessError::InvalidAttribute { ref name, .. } if name == "p"
        ));
    }

    #[test]
    fn infer_preserves_shape_and_dtype() {
        let mut node = TestNodeBuilder::new(NodeType::LpNormalization, "lpnorm")
            .input_tensor_f32("X", 3, Some(vec![2, 4, 5]))
            .output_tensor_f32("Y", 3, None)
            .build();
        let prefs = OutputPreferences::new();
        LpNormalizationProcessor
            .infer_types(&mut node, 1, &prefs)
            .unwrap();
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::F32);
                assert_eq!(t.rank, 3);
                assert_eq!(t.static_shape, Some(vec![Some(2), Some(4), Some(5)]));
            }
            _ => panic!("expected tensor output"),
        }
    }

    #[test]
    fn rejects_non_float_dtype() {
        let mut node = TestNodeBuilder::new(NodeType::LpNormalization, "lpnorm")
            .input_tensor_i32("X", 3, None)
            .output_tensor_f32("Y", 3, None)
            .build();
        let prefs = OutputPreferences::new();
        let err = LpNormalizationProcessor
            .infer_types(&mut node, 1, &prefs)
            .unwrap_err();
        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn rejects_bfloat16_before_opset_22() {
        let mut node = TestNodeBuilder::new(NodeType::LpNormalization, "lpnorm")
            .input_tensor_bf16("X", 3, None)
            .output_tensor_f32("Y", 3, None)
            .build();
        let prefs = OutputPreferences::new();
        let err = LpNormalizationProcessor
            .infer_types(&mut node, 1, &prefs)
            .unwrap_err();
        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn accepts_bfloat16_at_opset_22() {
        let mut node = TestNodeBuilder::new(NodeType::LpNormalization, "lpnorm")
            .input_tensor_bf16("X", 3, None)
            .output_tensor_f32("Y", 3, None)
            .build();
        let prefs = OutputPreferences::new();
        LpNormalizationProcessor
            .infer_types(&mut node, 22, &prefs)
            .unwrap();
    }
}
