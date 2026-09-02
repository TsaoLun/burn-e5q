// Import the shared macro
use crate::include_models;
include_models!(
    conv1d,
    conv1d_asymmetric_padding,
    conv1d_same_upper_dynamic,
    conv2d,
    conv2d_asymmetric_padding,
    conv2d_same_upper_dynamic,
    conv3d
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Shape, Tensor};
    use core::f64::consts;
    use float_cmp::ApproxEq;

    #[test]
    fn conv1d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv1d::Model = conv1d::Model::default();

        // Run the model with pi as input for easier testing
        let input = Tensor::<3>::full([6, 4, 10], consts::PI, &Default::default());

        let output = model.forward(input);

        // test the output shape
        let expected_shape: Shape = Shape::from([6, 2, 7]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv1d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum: f32 = -54.549_243; // from pytorch
        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn conv2d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv2d::Model = conv2d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<4>::ones([2, 4, 10, 15], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 6, 15]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv2d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar::<f32>();

        // PyTorch f32 ground truth (from conv2d/conv2d.py, torch 2.10.0 CPU).
        // Tolerance accommodates gemm-order accumulation drift on the ~1080-element
        // output sum; burn-flex's gemm-backed conv stays within ~2e-4 of PyTorch.
        let expected_sum = -113.86999_5_f32;
        assert!(expected_sum.approx_eq(output_sum, (1.0e-3, 2)));
    }

    #[test]
    fn conv3d() {
        // Initialize the model with weights (loaded from the exported file)
        let model: conv3d::Model = conv3d::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<5>::ones([2, 4, 4, 5, 7], &Default::default());

        let output = model.forward(input);

        let expected_shape = Shape::from([2, 6, 3, 5, 5]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness of the conv3d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar::<f32>();

        let expected_sum: f32 = 48.494_262; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn conv1d_asymmetric_padding() {
        // Initialize the model with weights (loaded from the exported file)
        // This model tests asymmetric padding: (left=1, right=2)
        let model: conv1d_asymmetric_padding::Model = conv1d_asymmetric_padding::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<3>::ones([2, 4, 10], &Default::default());

        let output = model.forward(input);

        // With asymmetric padding (1, 2), input length 10 becomes 10+1+2=13
        // After conv with kernel 3, stride 1, output length is 13-3+1=11
        let expected_shape = Shape::from([2, 6, 11]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness
        let output_sum = output.sum().into_scalar::<f32>();
        let expected_sum: f32 = -0.386_136; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-3, 2)));
    }

    #[test]
    fn conv2d_asymmetric_padding() {
        // Initialize the model with weights (loaded from the exported file)
        // This model tests asymmetric padding: (left=1, right=2, top=1, bottom=3)
        let model: conv2d_asymmetric_padding::Model = conv2d_asymmetric_padding::Model::default();

        // Run the model with ones as input for easier testing
        let input = Tensor::<4>::ones([2, 4, 10, 15], &Default::default());

        let output = model.forward(input);

        // With asymmetric padding (1, 2, 1, 3), input (10, 15) becomes (10+1+3, 15+1+2) = (14, 18)
        // After conv with kernel (3, 3), stride (1, 1), output is (12, 16)
        let expected_shape = Shape::from([2, 6, 12, 16]);
        assert_eq!(output.shape(), expected_shape);

        // We are using the sum of the output tensor to test the correctness
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum = output.sum().into_scalar::<f32>();

        // PyTorch f32 ground truth (from conv/conv2d_asymmetric_padding.py, torch 2.10.0
        // CPU). Tolerance is a bit larger than conv2d because the ~2304-element output
        // sum magnifies gemm-order accumulation drift (~2.5e-3 absolute vs PyTorch).
        let expected_sum = -481.67495_7_f32;
        assert!(expected_sum.approx_eq(output_sum, (5.0e-3, 2)));
    }

    #[test]
    fn conv1d_same_upper_dynamic() {
        // The 1D counterpart of conv2d_same_upper_dynamic. Its job is to compile and run
        // PaddingConfig1d::Same, which the snapshot tests only assert as generated text.
        // stride 2 makes the pads depend on the parity of the length: L=9 -> (1, 2), L=8 -> (1, 1).
        let device = Default::default();
        let model: conv1d_same_upper_dynamic::Model = conv1d_same_upper_dynamic::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/conv1d_same_upper_dynamic.bpk"),
            &device,
        );

        // Ground truth from onnxruntime (conv1d_same_upper_dynamic.py).
        for (length, out_len, expected_sum) in
            [(9usize, 5usize, -23.750_193_f32), (8, 4, -19.424_349)]
        {
            let input = Tensor::<3>::ones([1, 2, length], &device);
            let output = model.forward(input);

            assert_eq!(output.shape(), Shape::from([1, 3, out_len]));

            let output_sum = output.sum().into_scalar::<f32>();
            assert!(
                expected_sum.approx_eq(output_sum, (1.0e-3, 2)),
                "L={length}: expected {expected_sum}, got {output_sum}"
            );
        }
    }

    #[test]
    fn conv2d_same_upper_dynamic() {
        // auto_pad=SAME_UPPER over dynamic H/W: burn derives the pads from the real input size
        // at forward time. stride 2 makes the pads depend on the parity of the extent, so the
        // two sizes below exercise different pads (5x6 -> H (1,1) W (1,1); 8x7 -> H (0,1)
        // W (1,2)), and the odd totals exercise SAME_UPPER putting the extra pad last.
        let device = Default::default();
        let model: conv2d_same_upper_dynamic::Model = conv2d_same_upper_dynamic::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/conv2d_same_upper_dynamic.bpk"),
            &device,
        );

        // Ground truth from onnxruntime (see conv2d_same_upper_dynamic.py for why not
        // ReferenceEvaluator).
        for (height, width, out_h, out_w, expected_sum) in [
            (5usize, 6usize, 3usize, 3usize, -45.890_423_f32),
            (8, 7, 4, 4, -123.938_919),
        ] {
            let input = Tensor::<4>::ones([1, 2, height, width], &device);
            let output = model.forward(input);

            assert_eq!(output.shape(), Shape::from([1, 3, out_h, out_w]));

            let output_sum = output.sum().into_scalar::<f32>();
            assert!(
                expected_sum.approx_eq(output_sum, (1.0e-3, 2)),
                "{height}x{width}: expected {expected_sum}, got {output_sum}"
            );
        }
    }
}
