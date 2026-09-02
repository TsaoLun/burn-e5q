use crate::include_models;
include_models!(
    pad,
    pad_reflect,
    pad_edge,
    pad_runtime_constant,
    pad_runtime_pads,
    pad_runtime_pads_axes,
    pad_runtime_pads_shape,
    pad_runtime_axes,
    pad_optional_constant_value,
    pad_ndim
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn pad_constant() {
        let device = Default::default();
        let model: pad::Model = pad::Model::new(&device);

        let input = Tensor::<2>::from_floats([[1., 2.], [3., 4.], [5., 6.]], &device);
        let output = model.forward(input).to_data();
        let expected = TensorData::from([
            [0.0_f32, 0., 0., 0., 0., 0., 0., 0.],
            [0.0_f32, 0., 1., 2., 0., 0., 0., 0.],
            [0.0_f32, 0., 3., 4., 0., 0., 0., 0.],
            [0.0_f32, 0., 5., 6., 0., 0., 0., 0.],
            [0.0_f32, 0., 0., 0., 0., 0., 0., 0.],
            [0.0_f32, 0., 0., 0., 0., 0., 0., 0.],
            [0.0_f32, 0., 0., 0., 0., 0., 0., 0.],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad_reflect_mode() {
        let device = Default::default();
        let model: pad_reflect::Model = pad_reflect::Model::new(&device);

        // Input: 3x3 tensor
        let input = Tensor::<2>::from_floats([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], &device);
        let output = model.forward(input).to_data();

        // Expected with reflect padding (1,1,1,1):
        // Reflect excludes the edge value when mirroring
        let expected = TensorData::from([
            [5.0_f32, 4., 5., 6., 5.],
            [2.0_f32, 1., 2., 3., 2.],
            [5.0_f32, 4., 5., 6., 5.],
            [8.0_f32, 7., 8., 9., 8.],
            [5.0_f32, 4., 5., 6., 5.],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad_runtime_constant_value() {
        let device = Default::default();
        let model: pad_runtime_constant::Model = pad_runtime_constant::Model::new(&device);

        let input = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &device);
        let output = model.forward(input, 5.0_f32).to_data();
        let expected = TensorData::from([
            [5.0_f32, 5., 5., 5.],
            [5.0_f32, 1., 2., 5.],
            [5.0_f32, 3., 4., 5.],
            [5.0_f32, 5., 5., 5.],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad_runtime_pads() {
        let device = Default::default();
        let model: pad_runtime_pads::Model = pad_runtime_pads::Model::new(&device);

        let input = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &device);
        let pads = Tensor::<1, burn::tensor::Int>::from_ints([1_i64, 2, 1, 2], &device);
        let output = model.forward(input, pads).to_data();
        let expected = TensorData::from([
            [0.0_f32, 0., 0., 0., 0., 0.],
            [0.0_f32, 0., 1., 2., 0., 0.],
            [0.0_f32, 0., 3., 4., 0., 0.],
            [0.0_f32, 0., 0., 0., 0., 0.],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad_runtime_pads_axes() {
        let device = Default::default();
        let model: pad_runtime_pads_axes::Model = pad_runtime_pads_axes::Model::new(&device);

        let input = Tensor::<4>::from_data(
            TensorData::from([[[[1.0_f32, 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]]]),
            &device,
        );
        // pads layout for axes=[2, 0]:
        // [before_axis2, before_axis0, after_axis2, after_axis0]
        let pads = Tensor::<1, burn::tensor::Int>::from_ints([1_i64, 1, 2, 0], &device);

        let output = model.forward(input, pads);
        let dims = output.dims();
        assert_eq!(dims, [3, 1, 5, 2]);

        let data = output.to_data();
        let expected = TensorData::from([
            [[[0.0_f32, 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]],
            [[[0., 0.], [1., 2.], [3., 4.], [0., 0.], [0., 0.]]],
            [[[0., 0.], [5., 6.], [7., 8.], [0., 0.], [0., 0.]]],
        ]);
        data.assert_eq(&expected, true);
    }

    #[test]
    fn pad_runtime_axes() {
        // Both pads and axes supplied at runtime. Covers the
        // runtime-axes scatter path including negative-axis
        // normalization (a + rank).
        let device = Default::default();
        let model: pad_runtime_axes::Model = pad_runtime_axes::Model::new(&device);

        let input = Tensor::<4>::from_data(
            TensorData::from([[[[1.0_f32, 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]]]),
            &device,
        );
        let pads = Tensor::<1, burn::tensor::Int>::from_ints([1_i64, 1, 2, 0], &device);
        let axes = Tensor::<1, burn::tensor::Int>::from_ints([2_i64, 0], &device);

        let output = model.forward(input.clone(), pads.clone(), axes).to_data();
        let expected = TensorData::from([
            [[[0.0_f32, 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]],
            [[[0., 0.], [1., 2.], [3., 4.], [0., 0.], [0., 0.]]],
            [[[0., 0.], [5., 6.], [7., 8.], [0., 0.], [0., 0.]]],
        ]);
        output.assert_eq(&expected, true);

        // Negative axes: -2 == 2, -4 == 0 for rank-4 input.
        let neg_axes = Tensor::<1, burn::tensor::Int>::from_ints([-2_i64, -4], &device);
        let output2 = model.forward(input, pads, neg_axes).to_data();
        output2.assert_eq(&expected, true);
    }

    #[test]
    #[should_panic(expected = "out of range for rank 4")]
    fn pad_runtime_axes_out_of_range() {
        let device = Default::default();
        let model: pad_runtime_axes::Model = pad_runtime_axes::Model::new(&device);
        let input = Tensor::<4>::from_data(
            TensorData::from([[[[1.0_f32, 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]]]),
            &device,
        );
        let pads = Tensor::<1, burn::tensor::Int>::from_ints([1_i64, 1, 2, 0], &device);
        let bad_axes = Tensor::<1, burn::tensor::Int>::from_ints([7_i64, 0], &device);
        let _ = model.forward(input, pads, bad_axes);
    }

    #[test]
    #[should_panic(expected = "out of range for rank 4")]
    fn pad_runtime_axes_negative_overflow() {
        let device = Default::default();
        let model: pad_runtime_axes::Model = pad_runtime_axes::Model::new(&device);
        let input = Tensor::<4>::from_data(
            TensorData::from([[[[1.0_f32, 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]]]),
            &device,
        );
        let pads = Tensor::<1, burn::tensor::Int>::from_ints([1_i64, 1, 2, 0], &device);
        // -5 + 4 (rank) = -1, still out of range.
        let bad_axes = Tensor::<1, burn::tensor::Int>::from_ints([-5_i64, 0], &device);
        let _ = model.forward(input, pads, bad_axes);
    }

    #[test]
    #[should_panic(expected = "duplicate axis")]
    fn pad_runtime_axes_duplicate() {
        let device = Default::default();
        let model: pad_runtime_axes::Model = pad_runtime_axes::Model::new(&device);
        let input = Tensor::<4>::from_data(
            TensorData::from([[[[1.0_f32, 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]]]),
            &device,
        );
        let pads = Tensor::<1, burn::tensor::Int>::from_ints([1_i64, 1, 2, 0], &device);
        // 2 and -2 both normalize to dim 2 on rank 4.
        let dup_axes = Tensor::<1, burn::tensor::Int>::from_ints([2_i64, -2], &device);
        let _ = model.forward(input, pads, dup_axes);
    }

    #[test]
    #[should_panic(expected = "runtime pads length mismatch")]
    fn pad_runtime_axes_pads_length_mismatch() {
        let device = Default::default();
        let model: pad_runtime_axes::Model = pad_runtime_axes::Model::new(&device);
        let input = Tensor::<4>::from_data(
            TensorData::from([[[[1.0_f32, 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]]]),
            &device,
        );
        // axes.len() == 2 -> pads must be length 4; give 3 instead.
        let pads = Tensor::<1, burn::tensor::Int>::from_ints([1_i64, 1, 2], &device);
        let axes = Tensor::<1, burn::tensor::Int>::from_ints([2_i64, 0], &device);
        let _ = model.forward(input, pads, axes);
    }

    #[test]
    fn pad_runtime_pads_shape() {
        // pads is computed as Concat(Shape(a), Shape(b)) so the
        // simplifier classifies it as Shape(4). Exercises the
        // `[i64; N]`-indexing branch of the codegen.
        let device = Default::default();
        let model: pad_runtime_pads_shape::Model = pad_runtime_pads_shape::Model::new(&device);

        let data = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &device);
        // shape_a has dims (1, 2) → Shape = [1, 2]
        // shape_b has dims (2, 1) → Shape = [2, 1]
        // pads = [1, 2, 2, 1]
        let shape_a = Tensor::<2>::zeros([1, 2], &device);
        let shape_b = Tensor::<2>::zeros([2, 1], &device);

        let output = model.forward(data, shape_a, shape_b).to_data();
        let expected = TensorData::from([
            [0.0_f32, 0., 0., 0., 0.],
            [0., 0., 1., 2., 0.],
            [0., 0., 3., 4., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]);
        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad_optional_constant_value() {
        let device = Default::default();
        let model: pad_optional_constant_value::Model =
            pad_optional_constant_value::Model::new(&device);

        let input = Tensor::<2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let output = model.forward(input).to_data();
        let expected = TensorData::from([
            [0.0_f32, 0., 0., 0., 0.],
            [0.0_f32, 1., 2., 3., 0.],
            [0.0_f32, 4., 5., 6., 0.],
            [0.0_f32, 0., 0., 0., 0.],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad_ndim() {
        let device = Default::default();
        let model: pad_ndim::Model = pad_ndim::Model::new(&device);

        // Input: [1, 2, 3, 3] tensor with values 1..18
        let input = Tensor::<4>::from_data(
            TensorData::from([[
                [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]],
            ]]),
            &device,
        );
        let output = model.forward(input).to_data();

        // Pads: batch (1,0), channel (0,1), height (1,1), width (2,2)
        // Output shape: [2, 3, 5, 7]
        let expected = TensorData::from([
            [
                [
                    [0.0_f32, 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                ],
                [
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                ],
                [
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                ],
            ],
            [
                [
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 1., 2., 3., 0., 0.],
                    [0., 0., 4., 5., 6., 0., 0.],
                    [0., 0., 7., 8., 9., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                ],
                [
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 10., 11., 12., 0., 0.],
                    [0., 0., 13., 14., 15., 0., 0.],
                    [0., 0., 16., 17., 18., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                ],
                [
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                ],
            ],
        ]);

        output.assert_eq(&expected, true);
    }

    #[test]
    fn pad_edge_mode() {
        let device = Default::default();
        let model: pad_edge::Model = pad_edge::Model::new(&device);

        // Input: 2x3 tensor
        let input = Tensor::<2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let output = model.forward(input).to_data();

        // Expected with edge padding (1,1,1,1):
        // Edge replicates the boundary values
        let expected = TensorData::from([
            [1.0_f32, 1., 2., 3., 3.],
            [1.0_f32, 1., 2., 3., 3.],
            [4.0_f32, 4., 5., 6., 6.],
            [4.0_f32, 4., 5., 6., 6.],
        ]);

        output.assert_eq(&expected, true);
    }
}
