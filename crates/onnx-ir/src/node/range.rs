//! # Range
//!
//! Generates a tensor containing a sequence of numbers that begin at `start` and extends by
//! increments of `delta` up to `limit` (exclusive).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Range.html>
//!
//! ## Description
//!
//! The Range operator generates a 1-D tensor containing a sequence of evenly spaced values.
//! The generated sequence starts at `start` and increments by `delta` until reaching `limit`
//! (exclusive). This is similar to Python's `range()` function or NumPy's `arange()`.
//!
//! The number of elements in the output is computed as:
//! `number_of_elements = max(ceil((limit - start) / delta), 0)`
//!
//! Note that `limit` is **exclusive** - the output will not include the limit value itself.
//!
//! ## Type Constraints
//!
//! - T: tensor(double), tensor(float), tensor(int16), tensor(int32), tensor(int64)
//!
//! ## Opset Versions
//!
//! - **Opset 11**: Initial version with scalar inputs for start, limit, and delta.
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::ir::{ArgType, Node, RawNode, RuntimeInputRef, TensorDataExt, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for the Range operation.
#[derive(Debug, Clone, new)]
pub struct RangeConfig {
    pub start: RangeInput,
    pub limit: RangeInput,
    pub delta: RangeInput,
}

/// Represents either a static value or a runtime argument for range parameters.
#[derive(Debug, Clone)]
pub enum RangeInput {
    /// Static value known at compile time.
    Static(i64),
    /// Runtime argument determined during execution .
    Runtime(RuntimeInputRef),
}

impl Default for RangeInput {
    fn default() -> Self {
        Self::Static(0)
    }
}

/// Node representation for Range operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct RangeNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: RangeConfig,
}

pub(crate) struct RangeProcessor;

impl NodeProcessor for RangeProcessor {
    type Config = RangeConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Exact(3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn input_preferences(
        &self,
        node: &RawNode,
        _opset: usize,
    ) -> Result<Option<crate::processor::InputPreferences>, ProcessError> {
        use crate::processor::{ArgPreference, InputPreferences};

        // Range needs native scalars for arange bounds (start, limit, delta)
        let mut prefs = InputPreferences::new();
        for input in &node.inputs {
            prefs = prefs.add(&input.name, ArgPreference::ScalarNative);
        }
        Ok(Some(prefs))
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Only lift inputs that have static values
        // Runtime inputs (no value) should remain in the graph
        if !node.inputs.is_empty() && node.inputs[0].is_constant() {
            node.inputs[0].to_static()?;
        }

        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate that all three inputs have the same dtype (ONNX type T constraint)
        let start_dtype = node.inputs[0].ty.elem_type();
        for (i, name) in [(1, "limit"), (2, "delta")] {
            let dtype = node.inputs[i].ty.elem_type();
            if dtype != start_dtype {
                return Err(ProcessError::TypeMismatch {
                    expected: format!("{:?} (same as start)", start_dtype),
                    actual: format!("{} has dtype {:?}", name, dtype),
                });
            }
        }

        // Validate that all inputs are scalars (ONNX requires 0-D or 1-element 1-D)
        for input in &node.inputs {
            let is_valid = input.ty.is_scalar()
                || matches!(&input.ty, ArgType::Tensor(t) if t.rank == 0)
                || matches!(&input.ty, ArgType::Tensor(t) if t.rank == 1
                    && t.static_shape.as_ref().is_some_and(|s| s == &[Some(1)]));
            if !is_valid {
                return Err(ProcessError::TypeMismatch {
                    expected: "scalar input (0-D or 1-element 1-D tensor)".to_string(),
                    actual: format!("{} has type {:?}", input.name, input.ty),
                });
            }
        }

        let output_dtype = start_dtype;

        // Range operation always produces rank 1 tensor
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: output_dtype,
            rank: 1,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Helper function to extract range input
        fn get_range_input(
            node: &RawNode,
            index: usize,
            param_name: &str,
        ) -> Result<RangeInput, ProcessError> {
            let input = node.inputs.get(index).ok_or_else(|| {
                ProcessError::MissingInput(format!("Range: {} parameter is required", param_name))
            })?;

            match input.value() {
                None => Ok(RangeInput::Runtime(RuntimeInputRef::new(
                    input.name.clone(),
                    index,
                ))),
                Some(tensor_data) => match tensor_data.scalar_i64() {
                    Ok(value) => Ok(RangeInput::Static(value)),
                    Err(_) => Err(ProcessError::TypeMismatch {
                        expected: "scalar int value".to_string(),
                        actual: format!("{} must be a scalar int value", param_name),
                    }),
                },
            }
        }

        let start = get_range_input(node, 0, "start")?;
        let limit = get_range_input(node, 1, "limit")?;
        let delta = get_range_input(node, 2, "delta")?;

        // Reject delta=0 (causes division by zero in element count formula)
        if let RangeInput::Static(0) = &delta {
            return Err(ProcessError::InvalidAttribute {
                name: "delta".to_string(),
                reason: "delta must not be zero".to_string(),
            });
        }

        let config = RangeConfig {
            start,
            limit,
            delta,
        };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Range(RangeNode {
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
    use crate::ir::DType;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node() -> RawNode {
        TestNodeBuilder::new(NodeType::Range, "test_range")
            .input_scalar_i64("start")
            .input_scalar_i64("limit")
            .input_scalar_i64("delta")
            .output_tensor_i64("output", 0, None) // Rank 0 will be updated
            .build()
    }

    #[test]
    fn test_range_output() {
        let mut node = create_test_node();
        let processor = RangeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::I64);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_range_missing_inputs() {
        let mut node = create_test_node();
        node.inputs.pop();
        let processor = RangeProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_range_dtype_mismatch() {
        let node = TestNodeBuilder::new(NodeType::Range, "test_range")
            .input_scalar_i64("start")
            .input_scalar_f32("limit") // mismatched
            .input_scalar_i64("delta")
            .output_tensor_i64("output", 0, None)
            .build();
        let mut node = node;
        let processor = RangeProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_range_non_scalar_input_rank2() {
        let node = TestNodeBuilder::new(NodeType::Range, "test_range")
            .input_tensor_i64("start", 2, None) // rank 2, not scalar
            .input_scalar_i64("limit")
            .input_scalar_i64("delta")
            .output_tensor_i64("output", 0, None)
            .build();
        let mut node = node;
        let processor = RangeProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_range_non_scalar_input_rank1_multi_element() {
        // Rank 1 with shape [5] is not a valid scalar for Range
        let node = TestNodeBuilder::new(NodeType::Range, "test_range")
            .input_tensor_i64("start", 1, Some(vec![5]))
            .input_scalar_i64("limit")
            .input_scalar_i64("delta")
            .output_tensor_i64("output", 0, None)
            .build();
        let mut node = node;
        let processor = RangeProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_range_delta_zero() {
        let node = TestNodeBuilder::new(NodeType::Range, "test_range")
            .input_scalar_tensor_i64("start", Some(0))
            .input_scalar_tensor_i64("limit", Some(10))
            .input_scalar_tensor_i64("delta", Some(0))
            .output_tensor_i64("output", 0, None)
            .build_with_graph_data(16);
        let processor = RangeProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(
            matches!(result, Err(ProcessError::InvalidAttribute { .. })),
            "delta=0 should be rejected, got: {:?}",
            result
        );
    }
}
