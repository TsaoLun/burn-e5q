use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

#[derive(Debug, Clone, Default)]
pub struct DynamicQuantizeLinearConfig;

#[derive(Debug, Clone, NodeBuilder)]
pub struct DynamicQuantizeLinearNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: DynamicQuantizeLinearConfig,
}

pub(crate) struct DynamicQuantizeLinearProcessor;

impl NodeProcessor for DynamicQuantizeLinearProcessor {
    type Config = DynamicQuantizeLinearConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(3),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        if !node.inputs[0].ty.is_on_device() {
            return Err(ProcessError::TypeMismatch {
                expected: "on-device tensor for x".to_string(),
                actual: format!("{:?}", node.inputs[0].ty),
            });
        }

        let x_dtype = node.inputs[0].ty.elem_type();
        if !x_dtype.is_float() {
            return Err(ProcessError::TypeMismatch {
                expected: "float tensor for x".to_string(),
                actual: format!("{:?}", x_dtype),
            });
        }

        // Output y: same shape as x, uint8
        node.outputs[0].ty = match node.inputs[0].ty.clone() {
            ArgType::Tensor(tensor) => ArgType::Tensor(TensorType {
                dtype: DType::U8,
                rank: tensor.rank,
                static_shape: tensor.static_shape,
            }),
            ArgType::ScalarTensor(_) => ArgType::ScalarTensor(DType::U8),
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "tensor/scalar tensor input".to_string(),
                    actual: format!("{:?}", other),
                });
            }
        };

        // Output y_scale: scalar f32
        node.outputs[1].ty = ArgType::ScalarTensor(DType::F32);

        // Output y_zero_point: scalar u8
        node.outputs[2].ty = ArgType::ScalarTensor(DType::U8);

        Ok(())
    }

    fn extract_config(&self, _node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        Ok(DynamicQuantizeLinearConfig)
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::DynamicQuantizeLinear(DynamicQuantizeLinearNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config: DynamicQuantizeLinearConfig,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    #[test]
    fn test_dynamic_quantize_linear_infer_types() {
        let mut node = TestNodeBuilder::new(NodeType::DynamicQuantizeLinear, "dql")
            .input_tensor_f32("x", 2, None)
            .output_tensor_f32("y", 2, None)
            .output_tensor_f32("y_scale", 0, None)
            .output_tensor_f32("y_zero_point", 0, None)
            .build();

        let processor = DynamicQuantizeLinearProcessor;
        processor
            .infer_types(&mut node, 13, &OutputPreferences::new())
            .unwrap();

        assert_eq!(node.outputs[0].ty.elem_type(), DType::U8);
        assert_eq!(node.outputs[1].ty.elem_type(), DType::F32);
        assert_eq!(node.outputs[2].ty.elem_type(), DType::U8);
    }
}
