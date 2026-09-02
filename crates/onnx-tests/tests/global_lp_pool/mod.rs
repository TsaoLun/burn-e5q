use crate::include_models;
include_models!(
    global_lp_pool_default,
    global_lp_pool_l1,
    global_lp_pool_l2,
    global_lp_pool_l3,
    global_lp_pool_rank_4_l1,
    global_lp_pool_rank_4_l2,
    global_lp_pool_rank_4_l3,
    global_lp_pool_opset1_fractional_p
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    // Input generated via np.random.seed(42), shape [2, 3, 4]. Shared by the rank-3
    // tests so every p operates on the same data.
    fn test_input_rank_3(device: &Device) -> Tensor<3> {
        Tensor::<3>::from_floats(
            [
                [
                    [0.49671414, -0.13826430, 0.64768857, 1.52302980],
                    [-0.23415338, -0.23413695, 1.57921280, 0.76743470],
                    [-0.46947438, 0.54256004, -0.46341768, -0.46572974],
                ],
                [
                    [0.24196227, -1.91328020, -1.72491790, -0.56228750],
                    [-1.01283110, 0.31424734, -0.90802410, -1.41230370],
                    [1.46564880, -0.22577630, 0.06752820, -1.42474820],
                ],
            ],
            device,
        )
    }

    // Input generated via np.random.seed(42), shape [2, 3, 2, 3]. Shared by the
    // rank-4 tests.
    fn test_input_rank_4(device: &Device) -> Tensor<4> {
        Tensor::<4>::from_floats(
            [
                [
                    [
                        [0.49671414, -0.1382643, 0.64768857],
                        [1.5230298, -0.23415338, -0.23413695],
                    ],
                    [
                        [1.5792128, 0.7674347, -0.46947438],
                        [0.54256004, -0.46341768, -0.46572974],
                    ],
                    [
                        [0.24196227, -1.9132802, -1.7249179],
                        [-0.5622875, -1.0128311, 0.31424734],
                    ],
                ],
                [
                    [
                        [-0.9080241, -1.4123037, 1.4656488],
                        [-0.2257763, 0.0675282, -1.4247482],
                    ],
                    [
                        [-0.54438275, 0.11092259, -1.1509936],
                        [0.37569803, -0.6006387, -0.29169375],
                    ],
                    [
                        [-0.6017066, 1.8522782, -0.01349723],
                        [-1.0577109, 0.82254493, -1.2208437],
                    ],
                ],
            ],
            device,
        )
    }

    // `p` defaults to 2, so this asserts the same values as `global_lp_pool_l2`.
    // What it actually covers is that the attribute is optional.
    #[test]
    fn global_lp_pool_default() {
        let device = Default::default();
        let model: global_lp_pool_default::Model = global_lp_pool_default::Model::default();

        let output = model.forward(test_input_rank_3(&device));

        let expected = TensorData::from([
            [[1.7334827], [1.7867616], [0.9728503]],
            [[2.6477718], [1.985872], [2.0575638]],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn global_lp_pool_l1() {
        let device = Default::default();
        let model: global_lp_pool_l1::Model = global_lp_pool_l1::Model::default();

        let output = model.forward(test_input_rank_3(&device));

        let expected = TensorData::from([
            [[2.8056967], [2.8149376], [1.9411818]],
            [[4.4424477], [3.6474063], [3.1837015]],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn global_lp_pool_l2() {
        let device = Default::default();
        let model: global_lp_pool_l2::Model = global_lp_pool_l2::Model::default();

        let output = model.forward(test_input_rank_3(&device));

        let expected = TensorData::from([
            [[1.7334827], [1.7867616], [0.9728503]],
            [[2.6477718], [1.985872], [2.0575638]],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn global_lp_pool_l3() {
        let device = Default::default();
        let model: global_lp_pool_l3::Model = global_lp_pool_l3::Model::default();

        let output = model.forward(test_input_rank_3(&device));
        let expected = TensorData::from([
            [[1.5780534], [1.6406361], [0.77402496]],
            [[2.3101003], [1.6673921], [1.8223873]],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    // Opset 1 declares `p` as FLOAT with no integrality constraint. Neither
    // onnxruntime nor onnx.reference can execute GlobalLpPool(1), so these values come
    // from the spec formula, sum(|x|^p)^(1/p), computed in NumPy.
    #[test]
    fn global_lp_pool_opset1_fractional_p() {
        let device = Default::default();
        let model: global_lp_pool_opset1_fractional_p::Model =
            global_lp_pool_opset1_fractional_p::Model::default();

        let output = model.forward(test_input_rank_3(&device));
        let expected = TensorData::from([
            [[1.6279165], [1.6881945], [0.84793204]],
            [[2.4333966], [1.7827995], [1.9110897]],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn global_lp_pool_input_rank_4_l1() {
        let device = Default::default();
        let model: global_lp_pool_rank_4_l1::Model = global_lp_pool_rank_4_l1::Model::default();

        let output = model.forward(test_input_rank_4(&device));
        let expected = TensorData::from([
            [[[3.273987]], [[4.2878294]], [[5.769526]]],
            [[[5.5040293]], [[3.0743294]], [[5.5685816]]],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn global_lp_pool_input_rank_4_l2() {
        let device = Default::default();
        let model: global_lp_pool_rank_4_l2::Model = global_lp_pool_rank_4_l2::Model::default();

        let output = model.forward(test_input_rank_4(&device));
        let expected = TensorData::from([
            [[[1.7648258]], [[2.0073133]], [[2.85224]]],
            [[[2.6556878]], [[1.4901153]], [[2.6606314]]],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn global_lp_pool_input_rank_4_l3() {
        let device = Default::default();
        let model: global_lp_pool_rank_4_l3::Model = global_lp_pool_rank_4_l3::Model::default();

        let output = model.forward(test_input_rank_4(&device));
        let expected = TensorData::from([
            [[[1.5814824]], [[1.6931832]], [[2.3750906]]],
            [[[2.1266432]], [[1.2561412]], [[2.1638978]]],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
