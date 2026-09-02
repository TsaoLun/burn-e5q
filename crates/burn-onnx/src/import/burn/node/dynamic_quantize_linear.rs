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

        let x_rank = x_arg.ty.rank();

        // DynamicQuantizeLinear only supports uint8 output (opset 11+).
        // y_scale = (max(0, max(x)) - min(0, min(x))) / 255
        // y_zero_point = clamp(round(0 - min(x) / y_scale), 0, 255)
        // y = clamp(round(x / y_scale) + y_zero_point, 0, 255)

        // Build unsqueeze dims for broadcasting scalar (rank-0) to x_rank
        let unsqueeze_dims: Vec<isize> = (0..x_rank as isize).collect();
        let unsqueeze_dims_ts = quote! { &[#(#unsqueeze_dims),*] };

        quote! {
            let x_min = (#x).clone().min();
            let x_max = (#x).clone().max();

            let zero = burn::tensor::Tensor::zeros_like(&x_min);
            let x_min_adj = x_min.clone().min_pair(zero.clone());
            let x_max_adj = x_max.clone().max_pair(zero.clone());

            let #y_scale = (x_max_adj - x_min_adj.clone()).div_scalar(255f32);

            let zero_point_float = (zero - x_min_adj).div(#y_scale.clone());
            let #y_zero_point = zero_point_float.round().clamp(0f32, 255f32).int().cast(burn::tensor::DType::U8);

            // Broadcast scalar scale/zero_point to x's rank
            let y_scale_broadcast = (#y_scale).clone().unsqueeze_dims(#unsqueeze_dims_ts);
            let y_zero_point_broadcast = (#y_zero_point).clone().unsqueeze_dims(#unsqueeze_dims_ts);

            let #y = ((#x).div(y_scale_broadcast))
                .round()
                .add(y_zero_point_broadcast.float().cast(burn::tensor::DType::F32))
                .clamp(0f32, 255f32)
                .int()
                .cast(burn::tensor::DType::U8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::dynamic_quantize_linear::{
        DynamicQuantizeLinearConfig, DynamicQuantizeLinearNode, DynamicQuantizeLinearNodeBuilder,
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
            let x_min = (x).clone().min();
            let x_max = (x).clone().max();
            let zero = burn::tensor::Tensor::zeros_like(&x_min);
            let x_min_adj = x_min.clone().min_pair(zero.clone());
            let x_max_adj = x_max.clone().max_pair(zero.clone());
            let dql_out2 = (x_max_adj - x_min_adj.clone()).div_scalar(255f32);
            let zero_point_float = (zero - x_min_adj).div(dql_out2.clone());
            let dql_out3 = zero_point_float
                .round()
                .clamp(0f32, 255f32)
                .int()
                .cast(burn::tensor::DType::U8);
            let y_scale_broadcast = (dql_out2).clone().unsqueeze_dims(&[0isize]);
            let y_zero_point_broadcast = (dql_out3).clone().unsqueeze_dims(&[0isize]);
            let dql_out1 = ((x).div(y_scale_broadcast))
                .round()
                .add(
                    y_zero_point_broadcast
                        .float()
                        .cast(burn::tensor::DType::F32),
                )
                .clamp(0f32, 255f32)
                .int()
                .cast(burn::tensor::DType::U8);
            let dql_out2 = (dql_out2).into_scalar::<f32>();
            let dql_out3 = (dql_out3).into_scalar::<u8>();
            (dql_out1, dql_out2, dql_out3)
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
            let x_min = (x).clone().min();
            let x_max = (x).clone().max();
            let zero = burn::tensor::Tensor::zeros_like(&x_min);
            let x_min_adj = x_min.clone().min_pair(zero.clone());
            let x_max_adj = x_max.clone().max_pair(zero.clone());
            let dql_out2 = (x_max_adj - x_min_adj.clone()).div_scalar(255f32);
            let zero_point_float = (zero - x_min_adj).div(dql_out2.clone());
            let dql_out3 = zero_point_float
                .round()
                .clamp(0f32, 255f32)
                .int()
                .cast(burn::tensor::DType::U8);
            let y_scale_broadcast = (dql_out2).clone().unsqueeze_dims(&[0isize, 1isize, 2isize]);
            let y_zero_point_broadcast = (dql_out3).clone().unsqueeze_dims(&[0isize, 1isize, 2isize]);
            let dql_out1 = ((x).div(y_scale_broadcast))
                .round()
                .add(
                    y_zero_point_broadcast
                        .float()
                        .cast(burn::tensor::DType::F32),
                )
                .clamp(0f32, 255f32)
                .int()
                .cast(burn::tensor::DType::U8);
            let dql_out2 = (dql_out2).into_scalar::<f32>();
            let dql_out3 = (dql_out3).into_scalar::<u8>();
            (dql_out1, dql_out2, dql_out3)
        }
        ");
    }
}
