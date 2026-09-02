//! # CastLike
//!
//! Casts the elements of a given input tensor to the same data type as a target tensor.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__CastLike.html>
//!
//! ## Opset Versions
//! - **Opset 15**: Initial version
//! - **Opset 19**: Added saturate attribute for float8 conversions
//! - **Opset 21**: Added round_mode attribute for float8e8m0 conversion
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, AttributeValue, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for CastLike operations
#[derive(Debug, Clone, new)]
pub struct CastLikeConfig {
    /// Target element type to cast to (derived from the target_type input's dtype)
    pub to: DType,
    /// The parameter defines how the conversion behaves if an input value is out of
    /// range of the destination type (opset 19+, for float8 conversions only)
    pub saturate: Option<i64>,
    /// Rounding mode for conversion to float8e8m0 (opset 21+)
    pub round_mode: Option<String>,
}

/// Node representation for CastLike operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct CastLikeNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: CastLikeConfig,
}

pub(crate) struct CastLikeProcessor;

impl NodeProcessor for CastLikeProcessor {
    type Config = CastLikeConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 15,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn is_noop(&self, node: &RawNode) -> bool {
        // CastLike is a no-op when input and output types are identical
        node.inputs[0].ty == node.outputs[0].ty
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let config = self.extract_config(node, opset)?;
        let elem_type = config.to;

        // Infer output type based on input[0] type and target dtype from config
        let input = &mut node.inputs[0];
        let output = &mut node.outputs[0];

        match input.ty.clone() {
            ArgType::Tensor(tensor) => {
                if tensor.rank == 0 {
                    // treat 0-dim tensor as scalar
                    output.ty = ArgType::ScalarNative(elem_type);
                    input.ty = ArgType::ScalarNative(tensor.dtype);
                } else {
                    output.ty = ArgType::Tensor(TensorType {
                        dtype: elem_type,
                        rank: tensor.rank,
                        static_shape: tensor.static_shape,
                    });
                }
            }
            ArgType::ScalarTensor(_) => output.ty = ArgType::ScalarTensor(elem_type),
            ArgType::ScalarNative(_) => output.ty = ArgType::ScalarNative(elem_type),
            ArgType::Shape(rank) => {
                if elem_type.is_float() || elem_type.is_bool() {
                    output.ty = ArgType::Tensor(TensorType {
                        dtype: elem_type,
                        rank: 1,
                        static_shape: Some(vec![Some(rank)]),
                    });
                } else {
                    output.ty = ArgType::Shape(rank);
                }
            }
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Get target_type input (input[1]) to derive the target dtype
        let target_input = node
            .get_input(1)
            .ok_or_else(|| ProcessError::MissingInput("target_type".to_string()))?;

        let elem_type = match &target_input.ty {
            ArgType::Tensor(t) => t.dtype,
            ArgType::ScalarTensor(d) | ArgType::ScalarNative(d) => *d,
            ArgType::Shape(_) => {
                return Err(ProcessError::InvalidAttribute {
                    name: "target_type".to_string(),
                    reason: "Shape inputs are not valid target types for CastLike".to_string(),
                });
            }
        };

        // Extract optional 'saturate' attribute (opset 19+, for float8 conversions)
        let saturate = node.attrs.get("saturate").and_then(|v| {
            if let AttributeValue::Int64(i) = v {
                Some(*i)
            } else {
                None
            }
        });

        // Extract optional 'round_mode' attribute (opset 21+, for float8e8m0 conversions)
        let round_mode = node.attrs.get("round_mode").and_then(|v| {
            if let AttributeValue::String(s) = v {
                Some(s.clone())
            } else {
                None
            }
        });

        Ok(CastLikeConfig::new(elem_type, saturate, round_mode))
    }

    fn build_node(&self, mut builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        // Drop input[1] (target_type) - only the dtype is needed at type-inference time,
        // not the runtime value. The target dtype is stored in config.to.
        builder.inputs.truncate(1);

        Node::CastLike(CastLikeNode {
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

    fn create_test_node(input_dtype: DType, target_dtype: DType, rank: usize) -> RawNode {
        TestNodeBuilder::new(NodeType::CastLike, "test_cast_like")
            .add_input(
                "input",
                ArgType::Tensor(TensorType {
                    dtype: input_dtype,
                    rank,
                    static_shape: None,
                }),
            )
            .add_input(
                "target_type",
                ArgType::Tensor(TensorType {
                    dtype: target_dtype,
                    rank,
                    static_shape: None,
                }),
            )
            .add_output(
                "output",
                ArgType::Tensor(TensorType {
                    dtype: input_dtype, // will be overwritten by infer_types
                    rank,
                    static_shape: None,
                }),
            )
            .build()
    }

    #[test]
    fn test_cast_like_config_extraction() {
        let node = create_test_node(DType::F32, DType::I64, 2);
        let processor = CastLikeProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.to, DType::I64);
        assert_eq!(config.saturate, None);
    }

    #[test]
    fn test_cast_like_infer_types_tensor() {
        let mut node = create_test_node(DType::F32, DType::I64, 2);
        let processor = CastLikeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I64);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_cast_like_infer_types_scalar() {
        // 0-rank tensor becomes scalar
        let mut node = TestNodeBuilder::new(NodeType::CastLike, "test_cast_like")
            .input_tensor_f32("input", 0, None)
            .add_input(
                "target_type",
                ArgType::Tensor(TensorType {
                    dtype: DType::I32,
                    rank: 1,
                    static_shape: None,
                }),
            )
            .output_tensor_f32("output", 0, None)
            .build();

        let processor = CastLikeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::ScalarNative(dtype) => {
                assert_eq!(*dtype, DType::I32);
            }
            _ => panic!("Expected scalar native output for 0-rank tensor"),
        }
    }

    #[test]
    fn test_cast_like_is_noop() {
        let mut node = create_test_node(DType::F32, DType::F32, 2);
        let processor = CastLikeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(processor.is_noop(&node));
    }

    #[test]
    fn test_cast_like_is_not_noop() {
        let mut node = create_test_node(DType::F32, DType::I64, 2);
        let processor = CastLikeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert!(!processor.is_noop(&node));
    }

    #[test]
    fn test_cast_like_build_node_drops_input1() {
        let mut node = create_test_node(DType::F32, DType::I64, 2);
        let processor = CastLikeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let built = processor.build_node(node, 16);
        // After build_node, input[1] should be dropped
        assert_eq!(built.inputs().len(), 1);
    }

    #[test]
    fn test_cast_like_shape_to_int() {
        let mut node = TestNodeBuilder::new(NodeType::CastLike, "test_cast_like")
            .input_shape("input", 3)
            .input_tensor_i64("target_type", 1, None)
            .output_shape("output", 3)
            .build();

        let processor = CastLikeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 3);
            }
            _ => panic!("Expected Shape output when casting Shape to int"),
        }
    }

    #[test]
    fn test_cast_like_shape_to_float() {
        let mut node = TestNodeBuilder::new(NodeType::CastLike, "test_cast_like")
            .input_shape("input", 3)
            .input_tensor_f32("target_type", 1, None)
            .output_shape("output", 3)
            .build();

        let processor = CastLikeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected 1D float tensor when casting Shape to float"),
        }
    }

    #[test]
    fn test_cast_like_with_saturate_attribute() {
        let node = TestNodeBuilder::new(NodeType::CastLike, "test_cast_like")
            .input_tensor_f32("input", 2, None)
            .input_tensor_i64("target_type", 2, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("saturate", 1)
            .build();

        let processor = CastLikeProcessor;
        let config = processor.extract_config(&node, 19).unwrap();
        assert_eq!(config.to, DType::I64);
        assert_eq!(config.saturate, Some(1));
        assert_eq!(config.round_mode, None);
    }

    #[test]
    fn test_cast_like_with_round_mode_attribute() {
        let node = TestNodeBuilder::new(NodeType::CastLike, "test_cast_like")
            .input_tensor_f32("input", 2, None)
            .input_tensor_i64("target_type", 2, None)
            .output_tensor_f32("output", 2, None)
            .attr_string("round_mode", "nearest")
            .build();

        let processor = CastLikeProcessor;
        let config = processor.extract_config(&node, 21).unwrap();
        assert_eq!(config.to, DType::I64);
        assert_eq!(config.round_mode, Some("nearest".to_string()));
    }
}
