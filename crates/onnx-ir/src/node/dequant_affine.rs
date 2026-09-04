//! # DequantAffine (Burn-specific)
//!
//! Affine dequant of an integer GEMM product:
//! `y = x.f32 * scale + bias`, optionally followed by GELU.
//!
//! Created by the PHASE 4b coalesce pass from
//! `Cast(int→float) → Mul(scale) → Add(bias) [→ Gelu]`.
//! Not an ONNX operator — same class of synthetic node as `Linear`.
//!
//! **Related ONNX**: MatMulInteger, Cast, Mul, Add, Gelu.

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, AttributeValue, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Whether the fused affine dequant also applies GELU.
#[derive(Debug, Clone, new)]
pub struct DequantAffineConfig {
    pub apply_gelu: bool,
}

/// Node representation for fused affine dequant.
#[derive(Debug, Clone, NodeBuilder)]
pub struct DequantAffineNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: DequantAffineConfig,
}

pub(crate) struct DequantAffineProcessor;

impl NodeProcessor for DequantAffineProcessor {
    type Config = DequantAffineConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        for key in node.attrs.keys() {
            if key != "apply_gelu" {
                return Err(ProcessError::InvalidAttribute {
                    name: key.clone(),
                    reason: format!("DequantAffine only accepts 'apply_gelu', found: {key}"),
                });
            }
        }

        let tensor = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t,
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::F32,
            rank: tensor.rank,
            static_shape: tensor.static_shape.clone(),
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let apply_gelu = node
            .attrs
            .get("apply_gelu")
            .map(|v| matches!(v, AttributeValue::Int64(1)))
            .unwrap_or(false);
        Ok(DequantAffineConfig { apply_gelu })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::DequantAffine(DequantAffineNode {
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

    #[test]
    fn infer_types_is_f32_same_shape() {
        let mut node = TestNodeBuilder::new(NodeType::DequantAffine, "dq")
            .input_tensor_i32("x", 3, None)
            .input_tensor_f32("scale", 3, None)
            .input_tensor_f32("bias", 3, None)
            .output_tensor_f32("y", 3, None)
            .attr_int("apply_gelu", 1)
            .build();
        let processor = DequantAffineProcessor;
        processor
            .infer_types(&mut node, 11, &OutputPreferences::new())
            .unwrap();
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::F32);
                assert_eq!(t.rank, 3);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }
}
