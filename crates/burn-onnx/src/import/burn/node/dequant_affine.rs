use super::prelude::*;
use onnx_ir::node::dequant_affine::DequantAffineNode;

impl NodeCodegen for DequantAffineNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let x_arg = self.inputs.first().unwrap();
        let scale_arg = self.inputs.get(1).unwrap();
        let bias_arg = self.inputs.get(2).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let x = scope.arg(x_arg);
        let scale = scope.arg(scale_arg);
        let bias = scope.arg(bias_arg);

        let x_rank = x_arg.ty.rank();
        let scale_bc =
            broadcast_helpers::leading_broadcast(quote! { #scale }, scale_arg.ty.rank(), x_rank);
        let bias_bc =
            broadcast_helpers::leading_broadcast(quote! { #bias }, bias_arg.ty.rank(), x_rank);

        let call = if self.config.apply_gelu {
            quote! { (#x).dequant_affine_gelu(#scale_bc, #bias_bc) }
        } else {
            quote! { (#x).dequant_affine(#scale_bc, #bias_bc) }
        };

        quote! {
            let #output = #call;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::dequant_affine::{DequantAffineConfig, DequantAffineNodeBuilder};

    #[test]
    fn test_dequant_affine_same_rank() {
        let node = DequantAffineNodeBuilder::new("dq1")
            .input_tensor("x", 3, DType::I32)
            .input_tensor("scale", 3, DType::F32)
            .input_tensor("bias", 3, DType::F32)
            .output_tensor("y", 3, DType::F32)
            .config(DequantAffineConfig { apply_gelu: false })
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(
            &self,
            x: Tensor<3, Int>,
            scale: Tensor<3>,
            bias: Tensor<3>,
        ) -> Tensor<3> {
            let y = (x).dequant_affine(scale, bias);
            y
        }
        ");
    }

    #[test]
    fn test_dequant_affine_gelu_broadcast() {
        let node = DequantAffineNodeBuilder::new("dqg")
            .input_tensor("x", 3, DType::I32)
            .input_tensor("scale", 1, DType::F32)
            .input_tensor("bias", 1, DType::F32)
            .output_tensor("y", 3, DType::F32)
            .config(DequantAffineConfig { apply_gelu: true })
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(
            &self,
            x: Tensor<3, Int>,
            scale: Tensor<1>,
            bias: Tensor<1>,
        ) -> Tensor<3> {
            let y = (x)
                .dequant_affine_gelu(
                    (scale).unsqueeze_dims(&[0isize, 1isize]),
                    (bias).unsqueeze_dims(&[0isize, 1isize]),
                );
            y
        }
        ");
    }
}
