use super::prelude::*;
use crate::burn::TensorKind;
use onnx_ir::scatter_elements::ScatterElementsReduction;

impl NodeCodegen for onnx_ir::scatter_elements::ScatterElementsNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let axis = self.config.axis.to_tokens();
        let data = scope.arg(self.inputs.first().unwrap());
        let indices = scope.arg(&self.inputs[1]);
        let updates = scope.arg(&self.inputs[2]);
        let output = arg_to_ident(self.outputs.first().unwrap());

        let data_arg = self.inputs.first().unwrap();
        let (data_kind, rank) = match &data_arg.ty {
            ArgType::Tensor(t) => (TensorKind::from(t.dtype), t.rank),
            _ => {
                let msg = format!(
                    "ScatterElements node '{}': data input must be a tensor",
                    self.name
                );
                return quote! { let #output = { compile_error!(#msg); unreachable!() }; };
            }
        };
        let rank_lit = rank.to_tokens();

        if matches!(data_kind, TensorKind::Bool)
            && !matches!(self.config.reduction, ScatterElementsReduction::None)
        {
            let msg = format!(
                "ScatterElements node '{}': {:?} reduction is not supported for bool tensors",
                self.name, self.config.reduction
            );
            return quote! { let #output = { compile_error!(#msg); unreachable!() }; };
        }

        let update_op = match &self.config.reduction {
            ScatterElementsReduction::None => quote! { burn::tensor::IndexingUpdateOp::Assign },
            ScatterElementsReduction::Add => quote! { burn::tensor::IndexingUpdateOp::Add },
            ScatterElementsReduction::Mul => quote! { burn::tensor::IndexingUpdateOp::Mul },
            ScatterElementsReduction::Max => quote! { burn::tensor::IndexingUpdateOp::Max },
            ScatterElementsReduction::Min => quote! { burn::tensor::IndexingUpdateOp::Min },
        };

        // ONNX allows indices down to `-dim_size` along the scatter axis, which burn's
        // indexing does not accept, so fold negatives before scattering. Indices outside
        // `[-dim_size, dim_size - 1]` are an error per the ONNX spec and stay unchecked
        // here; adding a guard would mean reading the indices back to the host on every
        // forward pass.
        let prologue = quote! {
            let __se_data = #data;
            let __se_axis_size = __se_data.dims()[#axis] as i64;
            let __se_indices = #indices.cast(burn::tensor::DType::I64);
            let __se_negative = __se_indices.clone().lower_elem(0i64);
            let __se_corrected = __se_indices.clone() + __se_axis_size;
            let __se_indices = __se_indices.mask_where(__se_negative, __se_corrected);
        };

        // Add is the only update op burn implements for element-wise `scatter` on every
        // backend, and the only reduction ONNX and burn both define for duplicate indices.
        // Staying on it keeps the common case off the coordinate-building path below.
        if matches!(self.config.reduction, ScatterElementsReduction::Add) {
            return quote! {
                let #output = {
                    #prologue
                    __se_data.scatter(#axis, __se_indices, #updates, #update_op)
                };
            };
        }

        // No backend implements Min or Max for element-wise `scatter`; Assign is ndarray
        // only and Mul is ndarray and flex, while cubecl and tch inherit the Add-only
        // default body (tracel-ai/burn#5522). Every numeric backend does implement all
        // five for `scatter_nd`. ScatterElements assigns
        //   output[p_0, .., p_{axis-1}, indices[p], p_{axis+1}, .., p_{r-1}] = updates[p]
        // for every p in the index shape, so materializing those coordinates as index
        // tuples turns it into a ScatterND. The non-axis columns are the row-major
        // coordinates of p, recovered from a flat arange.
        //
        // Duplicate indices fold sequentially on the CPU backends but race on cubecl,
        // which burn documents as undefined for Assign, Mul, Min and Max.
        let strides = if rank > 1 {
            quote! {
                let mut __se_strides = [1usize; #rank_lit];
                for __se_d in (0..#rank_lit - 1).rev() {
                    __se_strides[__se_d] = __se_strides[__se_d + 1] * __se_idx_dims[__se_d + 1];
                }
            }
        } else {
            quote! { let __se_strides = [1usize; #rank_lit]; }
        };

        let coordinates = quote! {
            #strides
            let __se_flat = Tensor::<1, Int>::arange(
                0..__se_n as i64,
                (&self.device, burn::tensor::DType::I64),
            );
            let mut __se_columns: alloc::vec::Vec<Tensor<2, Int>> =
                alloc::vec::Vec::with_capacity(#rank_lit);
            for __se_d in 0..#rank_lit {
                __se_columns.push(if __se_d == #axis {
                    __se_indices.clone().reshape([__se_n, 1])
                } else {
                    __se_flat
                        .clone()
                        .div_scalar(__se_strides[__se_d] as i64)
                        .remainder_scalar(__se_idx_dims[__se_d] as i64)
                        .reshape([__se_n, 1])
                });
            }
            let __se_coordinates = Tensor::cat(__se_columns, 1);
        };

        // An empty index tensor is a legal ONNX no-op, but `scatter_nd` rejects empty
        // indices and `reshape([0, ..])` would read the 0 as "keep the source dim".
        let scatter = if matches!(data_kind, TensorKind::Bool) {
            // `scatter_nd` panics for bool tensors, so round-trip through i64.
            quote! {
                let __se_updates = #updates.int().cast(burn::tensor::DType::I64);
                __se_data
                    .int()
                    .cast(burn::tensor::DType::I64)
                    .scatter_nd(__se_coordinates, __se_updates.reshape([__se_n]), #update_op)
                    .bool()
            }
        } else {
            quote! {
                __se_data.scatter_nd(__se_coordinates, #updates.reshape([__se_n]), #update_op)
            }
        };

        quote! {
            let #output = {
                #prologue
                let __se_idx_dims = __se_indices.dims();
                let __se_n: usize = __se_idx_dims.iter().product();
                if __se_n == 0 {
                    __se_data
                } else {
                    #coordinates
                    #scatter
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::scatter_elements::{
        ScatterElementsConfig, ScatterElementsNodeBuilder, ScatterElementsReduction,
    };

    #[test]
    fn test_scatter_elements_none() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let __se_data = data;
                let __se_axis_size = __se_data.dims()[0] as i64;
                let __se_indices = indices.cast(burn::tensor::DType::I64);
                let __se_negative = __se_indices.clone().lower_elem(0i64);
                let __se_corrected = __se_indices.clone() + __se_axis_size;
                let __se_indices = __se_indices.mask_where(__se_negative, __se_corrected);
                let __se_idx_dims = __se_indices.dims();
                let __se_n: usize = __se_idx_dims.iter().product();
                if __se_n == 0 {
                    __se_data
                } else {
                    let mut __se_strides = [1usize; 2];
                    for __se_d in (0..2 - 1).rev() {
                        __se_strides[__se_d] = __se_strides[__se_d + 1]
                            * __se_idx_dims[__se_d + 1];
                    }
                    let __se_flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..__se_n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut __se_columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        2,
                    );
                    for __se_d in 0..2 {
                        __se_columns
                            .push(
                                if __se_d == 0 {
                                    __se_indices.clone().reshape([__se_n, 1])
                                } else {
                                    __se_flat
                                        .clone()
                                        .div_scalar(__se_strides[__se_d] as i64)
                                        .remainder_scalar(__se_idx_dims[__se_d] as i64)
                                        .reshape([__se_n, 1])
                                },
                            );
                    }
                    let __se_coordinates = Tensor::cat(__se_columns, 1);
                    __se_data
                        .scatter_nd(
                            __se_coordinates,
                            updates.reshape([__se_n]),
                            burn::tensor::IndexingUpdateOp::Assign,
                        )
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_add() {
        let config = ScatterElementsConfig::new(1, ScatterElementsReduction::Add);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let __se_data = data;
                let __se_axis_size = __se_data.dims()[1] as i64;
                let __se_indices = indices.cast(burn::tensor::DType::I64);
                let __se_negative = __se_indices.clone().lower_elem(0i64);
                let __se_corrected = __se_indices.clone() + __se_axis_size;
                let __se_indices = __se_indices.mask_where(__se_negative, __se_corrected);
                __se_data.scatter(1, __se_indices, updates, burn::tensor::IndexingUpdateOp::Add)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_mul() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Mul);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let __se_data = data;
                let __se_axis_size = __se_data.dims()[0] as i64;
                let __se_indices = indices.cast(burn::tensor::DType::I64);
                let __se_negative = __se_indices.clone().lower_elem(0i64);
                let __se_corrected = __se_indices.clone() + __se_axis_size;
                let __se_indices = __se_indices.mask_where(__se_negative, __se_corrected);
                let __se_idx_dims = __se_indices.dims();
                let __se_n: usize = __se_idx_dims.iter().product();
                if __se_n == 0 {
                    __se_data
                } else {
                    let mut __se_strides = [1usize; 2];
                    for __se_d in (0..2 - 1).rev() {
                        __se_strides[__se_d] = __se_strides[__se_d + 1]
                            * __se_idx_dims[__se_d + 1];
                    }
                    let __se_flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..__se_n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut __se_columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        2,
                    );
                    for __se_d in 0..2 {
                        __se_columns
                            .push(
                                if __se_d == 0 {
                                    __se_indices.clone().reshape([__se_n, 1])
                                } else {
                                    __se_flat
                                        .clone()
                                        .div_scalar(__se_strides[__se_d] as i64)
                                        .remainder_scalar(__se_idx_dims[__se_d] as i64)
                                        .reshape([__se_n, 1])
                                },
                            );
                    }
                    let __se_coordinates = Tensor::cat(__se_columns, 1);
                    __se_data
                        .scatter_nd(
                            __se_coordinates,
                            updates.reshape([__se_n]),
                            burn::tensor::IndexingUpdateOp::Mul,
                        )
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_max() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Max);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let __se_data = data;
                let __se_axis_size = __se_data.dims()[0] as i64;
                let __se_indices = indices.cast(burn::tensor::DType::I64);
                let __se_negative = __se_indices.clone().lower_elem(0i64);
                let __se_corrected = __se_indices.clone() + __se_axis_size;
                let __se_indices = __se_indices.mask_where(__se_negative, __se_corrected);
                let __se_idx_dims = __se_indices.dims();
                let __se_n: usize = __se_idx_dims.iter().product();
                if __se_n == 0 {
                    __se_data
                } else {
                    let mut __se_strides = [1usize; 2];
                    for __se_d in (0..2 - 1).rev() {
                        __se_strides[__se_d] = __se_strides[__se_d + 1]
                            * __se_idx_dims[__se_d + 1];
                    }
                    let __se_flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..__se_n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut __se_columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        2,
                    );
                    for __se_d in 0..2 {
                        __se_columns
                            .push(
                                if __se_d == 0 {
                                    __se_indices.clone().reshape([__se_n, 1])
                                } else {
                                    __se_flat
                                        .clone()
                                        .div_scalar(__se_strides[__se_d] as i64)
                                        .remainder_scalar(__se_idx_dims[__se_d] as i64)
                                        .reshape([__se_n, 1])
                                },
                            );
                    }
                    let __se_coordinates = Tensor::cat(__se_columns, 1);
                    __se_data
                        .scatter_nd(
                            __se_coordinates,
                            updates.reshape([__se_n]),
                            burn::tensor::IndexingUpdateOp::Max,
                        )
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_min() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Min);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2>,
            indices: Tensor<2, Int>,
            updates: Tensor<2>,
        ) -> Tensor<2> {
            let output = {
                let __se_data = data;
                let __se_axis_size = __se_data.dims()[0] as i64;
                let __se_indices = indices.cast(burn::tensor::DType::I64);
                let __se_negative = __se_indices.clone().lower_elem(0i64);
                let __se_corrected = __se_indices.clone() + __se_axis_size;
                let __se_indices = __se_indices.mask_where(__se_negative, __se_corrected);
                let __se_idx_dims = __se_indices.dims();
                let __se_n: usize = __se_idx_dims.iter().product();
                if __se_n == 0 {
                    __se_data
                } else {
                    let mut __se_strides = [1usize; 2];
                    for __se_d in (0..2 - 1).rev() {
                        __se_strides[__se_d] = __se_strides[__se_d + 1]
                            * __se_idx_dims[__se_d + 1];
                    }
                    let __se_flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..__se_n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut __se_columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        2,
                    );
                    for __se_d in 0..2 {
                        __se_columns
                            .push(
                                if __se_d == 0 {
                                    __se_indices.clone().reshape([__se_n, 1])
                                } else {
                                    __se_flat
                                        .clone()
                                        .div_scalar(__se_strides[__se_d] as i64)
                                        .remainder_scalar(__se_idx_dims[__se_d] as i64)
                                        .reshape([__se_n, 1])
                                },
                            );
                    }
                    let __se_coordinates = Tensor::cat(__se_columns, 1);
                    __se_data
                        .scatter_nd(
                            __se_coordinates,
                            updates.reshape([__se_n]),
                            burn::tensor::IndexingUpdateOp::Min,
                        )
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_int() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::I64)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 2, DType::I64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<2, Int>,
            indices: Tensor<2, Int>,
            updates: Tensor<2, Int>,
        ) -> Tensor<2, Int> {
            let output = {
                let __se_data = data;
                let __se_axis_size = __se_data.dims()[0] as i64;
                let __se_indices = indices.cast(burn::tensor::DType::I64);
                let __se_negative = __se_indices.clone().lower_elem(0i64);
                let __se_corrected = __se_indices.clone() + __se_axis_size;
                let __se_indices = __se_indices.mask_where(__se_negative, __se_corrected);
                let __se_idx_dims = __se_indices.dims();
                let __se_n: usize = __se_idx_dims.iter().product();
                if __se_n == 0 {
                    __se_data
                } else {
                    let mut __se_strides = [1usize; 2];
                    for __se_d in (0..2 - 1).rev() {
                        __se_strides[__se_d] = __se_strides[__se_d + 1]
                            * __se_idx_dims[__se_d + 1];
                    }
                    let __se_flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..__se_n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut __se_columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        2,
                    );
                    for __se_d in 0..2 {
                        __se_columns
                            .push(
                                if __se_d == 0 {
                                    __se_indices.clone().reshape([__se_n, 1])
                                } else {
                                    __se_flat
                                        .clone()
                                        .div_scalar(__se_strides[__se_d] as i64)
                                        .remainder_scalar(__se_idx_dims[__se_d] as i64)
                                        .reshape([__se_n, 1])
                                },
                            );
                    }
                    let __se_coordinates = Tensor::cat(__se_columns, 1);
                    __se_data
                        .scatter_nd(
                            __se_coordinates,
                            updates.reshape([__se_n]),
                            burn::tensor::IndexingUpdateOp::Assign,
                        )
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_bool_none() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::None);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool(BoolStore::Native))
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("updates", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<1, Bool>,
            indices: Tensor<1, Int>,
            updates: Tensor<1, Bool>,
        ) -> Tensor<1, Bool> {
            let output = {
                let __se_data = data;
                let __se_axis_size = __se_data.dims()[0] as i64;
                let __se_indices = indices.cast(burn::tensor::DType::I64);
                let __se_negative = __se_indices.clone().lower_elem(0i64);
                let __se_corrected = __se_indices.clone() + __se_axis_size;
                let __se_indices = __se_indices.mask_where(__se_negative, __se_corrected);
                let __se_idx_dims = __se_indices.dims();
                let __se_n: usize = __se_idx_dims.iter().product();
                if __se_n == 0 {
                    __se_data
                } else {
                    let __se_strides = [1usize; 1];
                    let __se_flat = Tensor::<
                        1,
                        Int,
                    >::arange(0..__se_n as i64, (&self.device, burn::tensor::DType::I64));
                    let mut __se_columns: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        1,
                    );
                    for __se_d in 0..1 {
                        __se_columns
                            .push(
                                if __se_d == 0 {
                                    __se_indices.clone().reshape([__se_n, 1])
                                } else {
                                    __se_flat
                                        .clone()
                                        .div_scalar(__se_strides[__se_d] as i64)
                                        .remainder_scalar(__se_idx_dims[__se_d] as i64)
                                        .reshape([__se_n, 1])
                                },
                            );
                    }
                    let __se_coordinates = Tensor::cat(__se_columns, 1);
                    let __se_updates = updates.int().cast(burn::tensor::DType::I64);
                    __se_data
                        .int()
                        .cast(burn::tensor::DType::I64)
                        .scatter_nd(
                            __se_coordinates,
                            __se_updates.reshape([__se_n]),
                            burn::tensor::IndexingUpdateOp::Assign,
                        )
                        .bool()
                }
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_elements_bool_add_emits_compile_error() {
        let config = ScatterElementsConfig::new(0, ScatterElementsReduction::Add);
        let node = ScatterElementsNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool(BoolStore::Native))
            .input_tensor("indices", 1, DType::I64)
            .input_tensor("updates", 1, DType::Bool(BoolStore::Native))
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            data: Tensor<1, Bool>,
            indices: Tensor<1, Int>,
            updates: Tensor<1, Bool>,
        ) -> Tensor<1, Bool> {
            let output = {
                compile_error!(
                    "ScatterElements node 'scatter1': Add reduction is not supported for bool tensors"
                );
                unreachable!()
            };
            output
        }
        "#);
    }
}
