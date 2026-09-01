//! Minimal PoC for DynamicQuantizeLinear via burn-onnx.
//!
//! The generated model is the official ONNX node test `test_dynamicquantizelinear`.
//! It takes a float tensor `x` and returns `(y, y_scale, y_zero_point)`.
//!
//! Run with:
//!   cargo run -p dql-poc

pub mod dql_matmul {
    include!(concat!(env!("OUT_DIR"), "/model/dql_matmul.rs"));
}

use burn::prelude::*;

fn main() {
    let device = burn::prelude::Device::default();
    let weights_path = concat!(env!("OUT_DIR"), "/model/dql_matmul.bpk");
    let model: dql_matmul::Model = dql_matmul::Model::from_file(weights_path, &device);

    // Official test input: [0., 2., -3., -2.5, 1.34, 0.5]
    let x = Tensor::<1>::from_floats([0.0, 2.0, -3.0, -2.5, 1.34, 0.5], &device);
    println!("input: {:?}", x.clone().to_data());
    let (y, y_scale, y_zero_point) = model.forward(x);

    println!("y: {:?}", y.to_data());
    println!("y_scale: {:?}", y_scale);
    println!("y_zero_point: {:?}", y_zero_point);
}
