use super::prelude::*;

impl NodeCodegen for onnx_ir::dynamic_quantize_linear::DynamicQuantizeLinearNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let x_arg = self.inputs.first().unwrap();
        let x = scope.arg(x_arg);
        let y = arg_to_ident(self.outputs.first().unwrap());
        let y_scale = arg_to_ident(self.outputs.get(1).unwrap());
        let y_zero_point = arg_to_ident(self.outputs.get(2).unwrap());

        // Fused ONNX DynamicQuantizeLinear. The backend default expands the
        // min/max/round/clamp formula; flex overrides with a two-pass kernel.
        // Scale/zp stay rank-1 tensors here; graph.rs still `into_scalar`s
        // boundary rank-0 outputs.
        quote! {
            let (#y, #y_scale, #y_zero_point) = (#x).dynamic_quantize_linear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::dynamic_quantize_linear::{
        DynamicQuantizeLinearConfig, DynamicQuantizeLinearNodeBuilder,
    };

    #[test]
    fn test_dynamic_quantize_linear_forward_1d() {
        let node = DynamicQuantizeLinearNodeBuilder::new("dql")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::U8)
            .output_scalar("y_scale", DType::F32)
            .output_scalar("y_zero_point", DType::U8)
            .config(DynamicQuantizeLinearConfig::default())
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<1>) -> (Tensor<1, Int>, f32, u8) {
            let (y, y_scale, y_zero_point) = (x).dynamic_quantize_linear();
            (y, y_scale, y_zero_point)
        }
        ");
    }

    #[test]
    fn test_dynamic_quantize_linear_forward_3d() {
        let node = DynamicQuantizeLinearNodeBuilder::new("dql")
            .input_tensor("x", 3, DType::F32)
            .output_tensor("y", 3, DType::U8)
            .output_scalar("y_scale", DType::F32)
            .output_scalar("y_zero_point", DType::U8)
            .config(DynamicQuantizeLinearConfig::default())
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<3>) -> (Tensor<3, Int>, f32, u8) {
            let (y, y_scale, y_zero_point) = (x).dynamic_quantize_linear();
            (y, y_scale, y_zero_point)
        }
        ");
    }
}
