use super::prelude::*;

impl NodeCodegen for onnx_ir::matmulinteger::MatMulIntegerNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs = scope.arg(self.inputs.first().unwrap());
        let rhs = scope.arg(self.inputs.get(1).unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // MatMulInteger output is always I32. Zero-point correction uses this
        // dtype; the matmul itself keeps the input element types so an 8-bit
        // backend kernel (u8/i8 → i32) can run.
        let output_dtype = self.outputs.first().unwrap().ty.elem_type().to_tokens();

        let lhs_rank = match &self.inputs.first().unwrap().ty {
            onnx_ir::ir::ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for lhs"),
        };
        let rhs_rank = match &self.inputs.get(1).unwrap().ty {
            onnx_ir::ir::ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for rhs"),
        };

        let zp_a = self.inputs.get(2).map(|arg| scope.arg(arg));
        let zp_b = self.inputs.get(3).map(|arg| scope.arg(arg));

        // Rank-align without changing dtypes, then matmul, then (optional) zp
        // correction on the I32 product:
        //   (A-za)@(B-zb) = A@B − za·sum_k(B) − sum_k(A)·zb + za·zb·K
        // ONNX zero-points are constant along K (scalar or per-row / per-col),
        // so the sums are exact. Centering before matmul would widen to I32
        // and miss the 8-bit kernel.
        match lhs_rank.cmp(&rhs_rank) {
            std::cmp::Ordering::Greater => {
                let num_unsqueezes = lhs_rank - rhs_rank;
                if rhs_rank == 1 {
                    let squeeze_dim = lhs_rank - 1;
                    let out_rank = lhs_rank - 1;
                    let mut unsqueeze_dims = vec![-1isize];
                    if num_unsqueezes > 1 {
                        unsqueeze_dims.extend(std::iter::repeat_n(0isize, num_unsqueezes - 1));
                    }
                    let rhs_e = quote! { (#rhs).clone().unsqueeze_dims(&[#(#unsqueeze_dims),*]) };
                    let prod = integer_matmul_with_zp(
                        quote! { #lhs },
                        rhs_e,
                        lhs_rank,
                        zp_a.as_ref(),
                        zp_b.as_ref(),
                        &output_dtype,
                    );
                    quote! {
                        let #output = (#prod).squeeze_dim::<#out_rank>(#squeeze_dim);
                    }
                } else {
                    let target_rank = lhs_rank;
                    let rhs_e = quote! { (#rhs).clone().unsqueeze::<#target_rank>() };
                    let prod = integer_matmul_with_zp(
                        quote! { #lhs },
                        rhs_e,
                        target_rank,
                        zp_a.as_ref(),
                        zp_b.as_ref(),
                        &output_dtype,
                    );
                    quote! {
                        let #output = #prod;
                    }
                }
            }
            std::cmp::Ordering::Less => {
                if lhs_rank == 1 {
                    let squeeze_dim = rhs_rank - 2;
                    let out_rank = rhs_rank - 1;
                    let target_rank = rhs_rank;
                    let lhs_e = quote! { (#lhs).clone().unsqueeze::<#target_rank>() };
                    let prod = integer_matmul_with_zp(
                        lhs_e,
                        quote! { #rhs },
                        target_rank,
                        zp_a.as_ref(),
                        zp_b.as_ref(),
                        &output_dtype,
                    );
                    quote! {
                        let #output = (#prod).squeeze_dim::<#out_rank>(#squeeze_dim);
                    }
                } else {
                    let target_rank = rhs_rank;
                    let lhs_e = quote! { (#lhs).clone().unsqueeze::<#target_rank>() };
                    let prod = integer_matmul_with_zp(
                        lhs_e,
                        quote! { #rhs },
                        target_rank,
                        zp_a.as_ref(),
                        zp_b.as_ref(),
                        &output_dtype,
                    );
                    quote! {
                        let #output = #prod;
                    }
                }
            }
            std::cmp::Ordering::Equal => {
                let prod = integer_matmul_with_zp(
                    quote! { #lhs },
                    quote! { #rhs },
                    lhs_rank,
                    zp_a.as_ref(),
                    zp_b.as_ref(),
                    &output_dtype,
                );
                quote! {
                    let #output = #prod;
                }
            }
        }
    }
}

/// `lhs @ rhs` in the operands' native dtype, plus algebraic zero-point
/// correction in `output_dtype` (I32).
fn integer_matmul_with_zp(
    lhs: TokenStream,
    rhs: TokenStream,
    aligned_rank: usize,
    zp_a: Option<&TokenStream>,
    zp_b: Option<&TokenStream>,
    output_dtype: &TokenStream,
) -> TokenStream {
    if zp_a.is_none() && zp_b.is_none() {
        return quote! { (#lhs).matmul(#rhs) };
    }

    let k_lhs = aligned_rank.saturating_sub(1);
    let k_rhs = if aligned_rank >= 2 {
        aligned_rank - 2
    } else {
        0
    };

    // Every consuming use clones the source identifier. zp correction mentions
    // lhs/rhs/za/zb more than once; without this the generated model hits
    // use-after-move (e5 has 96 MatMulInteger nodes, almost all with zp).
    let zp_expr = |zp: &TokenStream| {
        if aligned_rank > 1 {
            quote! { (#zp).clone().cast(#output_dtype).unsqueeze::<#aligned_rank>() }
        } else {
            quote! { (#zp).clone().cast(#output_dtype) }
        }
    };

    let mut prod = quote! { (#lhs).clone().matmul((#rhs).clone()) };

    if let Some(zp) = zp_a {
        let za = zp_expr(zp);
        prod = quote! {
            #prod.sub((#za).mul((#rhs).clone().cast(#output_dtype).sum_dim(#k_rhs)))
        };
    }
    if let Some(zp) = zp_b {
        let zb = zp_expr(zp);
        prod = quote! {
            #prod.sub((#lhs).clone().cast(#output_dtype).sum_dim(#k_lhs).mul(#zb))
        };
    }
    if let (Some(za_src), Some(zb_src)) = (zp_a, zp_b) {
        let za = zp_expr(za_src);
        let zb = zp_expr(zb_src);
        prod = quote! {
            #prod.add((#za).mul(#zb).mul_scalar((#lhs).dims()[#k_lhs] as i32))
        };
    }
    prod
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::matmulinteger::MatMulIntegerNodeBuilder;

    #[test]
    fn test_matmul_integer_same_rank() {
        let node = MatMulIntegerNodeBuilder::new("mmint1")
            .input_tensor("a", 2, DType::I32)
            .input_tensor("b", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<2, Int>, b: Tensor<2, Int>) -> Tensor<2, Int> {
            let output = (a).matmul(b);
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_u8_i8_keeps_input_dtypes() {
        let node = MatMulIntegerNodeBuilder::new("mmint_u8i8")
            .input_tensor("a", 2, DType::U8)
            .input_tensor("b", 2, DType::I8)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<2, Int>, b: Tensor<2, Int>) -> Tensor<2, Int> {
            let output = (a).matmul(b);
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_with_zero_points() {
        let node = MatMulIntegerNodeBuilder::new("mmint2")
            .input_tensor("a", 2, DType::I32)
            .input_tensor("b", 2, DType::I32)
            .input_tensor("a_zero_point", 2, DType::I32)
            .input_tensor("b_zero_point", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<2, Int>,
            b: Tensor<2, Int>,
            a_zero_point: Tensor<2, Int>,
            b_zero_point: Tensor<2, Int>,
        ) -> Tensor<2, Int> {
            let output = (a)
                .clone()
                .matmul((b).clone())
                .sub(
                    ((a_zero_point).clone().cast(burn::tensor::DType::I32).unsqueeze::<2usize>())
                        .mul((b).clone().cast(burn::tensor::DType::I32).sum_dim(0usize)),
                )
                .sub(
                    (a)
                        .clone()
                        .cast(burn::tensor::DType::I32)
                        .sum_dim(1usize)
                        .mul(
                            (b_zero_point)
                                .clone()
                                .cast(burn::tensor::DType::I32)
                                .unsqueeze::<2usize>(),
                        ),
                )
                .add(
                    ((a_zero_point).clone().cast(burn::tensor::DType::I32).unsqueeze::<2usize>())
                        .mul(
                            (b_zero_point)
                                .clone()
                                .cast(burn::tensor::DType::I32)
                                .unsqueeze::<2usize>(),
                        )
                        .mul_scalar((a).dims()[1usize] as i32),
                );
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_lhs_zero_point_only() {
        let node = MatMulIntegerNodeBuilder::new("mmint3")
            .input_tensor("a", 2, DType::I32)
            .input_tensor("b", 2, DType::I32)
            .input_tensor("a_zero_point", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<2, Int>,
            b: Tensor<2, Int>,
            a_zero_point: Tensor<2, Int>,
        ) -> Tensor<2, Int> {
            let output = (a)
                .clone()
                .matmul((b).clone())
                .sub(
                    ((a_zero_point).clone().cast(burn::tensor::DType::I32).unsqueeze::<2usize>())
                        .mul((b).clone().cast(burn::tensor::DType::I32).sum_dim(0usize)),
                );
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_rank_mismatch_with_zero_points() {
        let node = MatMulIntegerNodeBuilder::new("mmint_e5")
            .input_tensor("a", 3, DType::U8)
            .input_tensor("b", 2, DType::I8)
            .input_tensor("a_zero_point", 1, DType::U8)
            .input_tensor("b_zero_point", 1, DType::I8)
            .output_tensor("output", 3, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<3, Int>,
            b: Tensor<2, Int>,
            a_zero_point: Tensor<1, Int>,
            b_zero_point: Tensor<1, Int>,
        ) -> Tensor<3, Int> {
            let output = (a)
                .clone()
                .matmul(((b).clone().unsqueeze::<3usize>()).clone())
                .sub(
                    ((a_zero_point).clone().cast(burn::tensor::DType::I32).unsqueeze::<3usize>())
                        .mul(
                            ((b).clone().unsqueeze::<3usize>())
                                .clone()
                                .cast(burn::tensor::DType::I32)
                                .sum_dim(1usize),
                        ),
                )
                .sub(
                    (a)
                        .clone()
                        .cast(burn::tensor::DType::I32)
                        .sum_dim(2usize)
                        .mul(
                            (b_zero_point)
                                .clone()
                                .cast(burn::tensor::DType::I32)
                                .unsqueeze::<3usize>(),
                        ),
                )
                .add(
                    ((a_zero_point).clone().cast(burn::tensor::DType::I32).unsqueeze::<3usize>())
                        .mul(
                            (b_zero_point)
                                .clone()
                                .cast(burn::tensor::DType::I32)
                                .unsqueeze::<3usize>(),
                        )
                        .mul_scalar((a).dims()[2usize] as i32),
                );
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_lhs_greater_rank() {
        let node = MatMulIntegerNodeBuilder::new("mmint4")
            .input_tensor("a", 3, DType::I32)
            .input_tensor("b", 2, DType::I32)
            .output_tensor("output", 3, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<3, Int>, b: Tensor<2, Int>) -> Tensor<3, Int> {
            let output = (a).matmul((b).clone().unsqueeze::<3usize>());
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_rhs_greater_rank() {
        let node = MatMulIntegerNodeBuilder::new("mmint5")
            .input_tensor("a", 2, DType::I32)
            .input_tensor("b", 3, DType::I32)
            .output_tensor("output", 3, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<2, Int>, b: Tensor<3, Int>) -> Tensor<3, Int> {
            let output = ((a).clone().unsqueeze::<3usize>()).matmul(b);
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_matrix_vector() {
        let node = MatMulIntegerNodeBuilder::new("mmint6")
            .input_tensor("a", 2, DType::I32)
            .input_tensor("b", 1, DType::I32)
            .output_tensor("output", 1, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<2, Int>, b: Tensor<1, Int>) -> Tensor<1, Int> {
            let output = ((a).matmul((b).clone().unsqueeze_dims(&[-1isize])))
                .squeeze_dim::<1usize>(1usize);
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_vector_matrix() {
        let node = MatMulIntegerNodeBuilder::new("mmint7")
            .input_tensor("a", 1, DType::I32)
            .input_tensor("b", 2, DType::I32)
            .output_tensor("output", 1, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<1, Int>, b: Tensor<2, Int>) -> Tensor<1, Int> {
            let output = (((a).clone().unsqueeze::<2usize>()).matmul(b))
                .squeeze_dim::<1usize>(0usize);
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_3d_vector() {
        let node = MatMulIntegerNodeBuilder::new("mmint8")
            .input_tensor("a", 3, DType::I32)
            .input_tensor("b", 1, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<3, Int>, b: Tensor<1, Int>) -> Tensor<2, Int> {
            let output = ((a).matmul((b).clone().unsqueeze_dims(&[-1isize, 0isize])))
                .squeeze_dim::<2usize>(2usize);
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_zero_points_scalar_rank1() {
        let node = MatMulIntegerNodeBuilder::new("mmint9")
            .input_tensor("a", 1, DType::I32)
            .input_tensor("b", 1, DType::I32)
            .input_tensor("a_zero_point", 1, DType::I32)
            .input_tensor("b_zero_point", 1, DType::I32)
            .output_tensor("output", 1, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<1, Int>,
            b: Tensor<1, Int>,
            a_zero_point: Tensor<1, Int>,
            b_zero_point: Tensor<1, Int>,
        ) -> Tensor<1, Int> {
            let output = (a)
                .clone()
                .matmul((b).clone())
                .sub(
                    ((a_zero_point).clone().cast(burn::tensor::DType::I32))
                        .mul((b).clone().cast(burn::tensor::DType::I32).sum_dim(0usize)),
                )
                .sub(
                    (a)
                        .clone()
                        .cast(burn::tensor::DType::I32)
                        .sum_dim(0usize)
                        .mul((b_zero_point).clone().cast(burn::tensor::DType::I32)),
                )
                .add(
                    ((a_zero_point).clone().cast(burn::tensor::DType::I32))
                        .mul((b_zero_point).clone().cast(burn::tensor::DType::I32))
                        .mul_scalar((a).dims()[0usize] as i32),
                );
            output
        }
        ");
    }
}
