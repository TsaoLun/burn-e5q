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

/// `lhs @ rhs` in the operands' native dtype. Zero-points are passed through
/// to `Tensor::matmul_integer` so a backend can fuse
/// `(A-za)@(B-zb)` into one GEMM (flex VNNI does). Each zp identifier is
/// cloned — e5 reuses a DQL zp across several MatMulInteger nodes.
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

    let zp_arg = |zp: Option<&TokenStream>| match zp {
        None => quote! { None },
        Some(zp) if aligned_rank > 1 => {
            quote! { Some((#zp).clone().cast(#output_dtype).unsqueeze::<#aligned_rank>()) }
        }
        Some(zp) => quote! { Some((#zp).clone().cast(#output_dtype)) },
    };
    let za = zp_arg(zp_a);
    let zb = zp_arg(zp_b);
    quote! { (#lhs).matmul_integer(#rhs, #za, #zb) }
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
                .matmul_integer(
                    b,
                    Some(
                        (a_zero_point)
                            .clone()
                            .cast(burn::tensor::DType::I32)
                            .unsqueeze::<2usize>(),
                    ),
                    Some(
                        (b_zero_point)
                            .clone()
                            .cast(burn::tensor::DType::I32)
                            .unsqueeze::<2usize>(),
                    ),
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
                .matmul_integer(
                    b,
                    Some(
                        (a_zero_point)
                            .clone()
                            .cast(burn::tensor::DType::I32)
                            .unsqueeze::<2usize>(),
                    ),
                    None,
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
                .matmul_integer(
                    (b).clone().unsqueeze::<3usize>(),
                    Some(
                        (a_zero_point)
                            .clone()
                            .cast(burn::tensor::DType::I32)
                            .unsqueeze::<3usize>(),
                    ),
                    Some(
                        (b_zero_point)
                            .clone()
                            .cast(burn::tensor::DType::I32)
                            .unsqueeze::<3usize>(),
                    ),
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
                .matmul_integer(
                    b,
                    Some((a_zero_point).clone().cast(burn::tensor::DType::I32)),
                    Some((b_zero_point).clone().cast(burn::tensor::DType::I32)),
                );
            output
        }
        ");
    }
}
