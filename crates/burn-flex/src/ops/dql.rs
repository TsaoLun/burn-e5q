//! Fused ONNX DynamicQuantizeLinear for flex.
//!
//! One minmax pass + one quantize pass (ties-to-even), instead of the
//! ~10-op expansion (`min`/`max`/`div`/`round`/`add`/`clamp`/`cast`).

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, TensorData, TensorMetadata, ops::FloatTensorOps};
use burn_std::Shape;

use crate::ops::float_storage_as_f32;
use crate::{Flex, FlexTensor};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Chunk size for rayon fan-out on the minmax / quantize loops.
const CHUNK: usize = 16 * 1024;

pub(crate) fn dynamic_quantize_linear(tensor: FlexTensor) -> (FlexTensor, FlexTensor, FlexTensor) {
    let tensor = if tensor.dtype() == DType::F32 {
        tensor.to_contiguous()
    } else {
        Flex::float_cast(tensor, burn_std::FloatDType::F32).to_contiguous()
    };
    let shape = tensor.shape();
    let x = float_storage_as_f32(&tensor);
    let (y, scale, zp) = dql_u8(&x);
    let y_t = FlexTensor::from_data(TensorData::new(y, shape));
    let scale_t = FlexTensor::from_data(TensorData::new(vec![scale], Shape::from(vec![1])));
    let zp_t = FlexTensor::from_data(TensorData::new(vec![zp], Shape::from(vec![1])));
    (y_t, scale_t, zp_t)
}

pub(crate) fn dql_u8(x: &[f32]) -> (Vec<u8>, f32, u8) {
    if x.is_empty() {
        return (Vec::new(), 0.0, 0);
    }

    let (xmin, xmax) = minmax(x);
    let xmin_adj = xmin.min(0.0);
    let xmax_adj = xmax.max(0.0);
    let scale = (xmax_adj - xmin_adj) / 255.0;

    if !(scale > 0.0) || !scale.is_finite() {
        return (
            vec![0u8; x.len()],
            if scale.is_finite() { scale } else { 0.0 },
            0,
        );
    }

    let zp = (0.0 - xmin_adj / scale).round_ties_even().clamp(0.0, 255.0) as u8;
    let zp_f = zp as f32;
    let y = quantize(x, scale, zp_f);
    (y, scale, zp)
}

fn minmax(x: &[f32]) -> (f32, f32) {
    #[cfg(feature = "rayon")]
    {
        if x.len() >= crate::ops::PARALLEL_THRESHOLD {
            return x.par_chunks(CHUNK).map(minmax_slice).reduce(
                || (f32::INFINITY, f32::NEG_INFINITY),
                |a, b| (a.0.min(b.0), a.1.max(b.1)),
            );
        }
    }
    minmax_slice(x)
}

fn minmax_slice(x: &[f32]) -> (f32, f32) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in x {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    (mn, mx)
}

fn quantize(x: &[f32], scale: f32, zp: f32) -> Vec<u8> {
    let mut y = vec![0u8; x.len()];
    #[cfg(feature = "rayon")]
    {
        if x.len() >= crate::ops::PARALLEL_THRESHOLD {
            y.par_iter_mut()
                .zip(x.par_iter())
                .for_each(|(o, &v)| *o = quant_one(v, scale, zp));
            return y;
        }
    }
    for (o, &v) in y.iter_mut().zip(x) {
        *o = quant_one(v, scale, zp);
    }
    y
}

#[inline(always)]
fn quant_one(v: f32, scale: f32, zp: f32) -> u8 {
    ((v / scale).round_ties_even() + zp).clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::ops::FloatTensorOps;

    fn run_fused(vals: Vec<f32>, dims: &[usize]) -> (Vec<u8>, f32, u8) {
        let t = FlexTensor::from_data(TensorData::new(vals, Shape::from(dims.to_vec())));
        let (y, scale, zp) = Flex::float_dynamic_quantize_linear(t);
        let yv: Vec<u8> = y.into_data().try_into_vec().unwrap();
        let sv: Vec<f32> = scale.into_data().try_into_vec().unwrap();
        let zv: Vec<u8> = zp.into_data().try_into_vec().unwrap();
        (yv, sv[0], zv[0])
    }

    fn run_expanded(vals: Vec<f32>, dims: &[usize]) -> (Vec<u8>, f32, u8) {
        let t = FlexTensor::from_data(TensorData::new(vals, Shape::from(dims.to_vec())));
        let (y, scale, zp) = burn_backend::ops::float_dynamic_quantize_linear_expanded::<Flex>(t);
        let yv: Vec<u8> = y.into_data().try_into_vec().unwrap();
        let sv: Vec<f32> = scale.into_data().try_into_vec().unwrap();
        let zv: Vec<u8> = zp.into_data().try_into_vec().unwrap();
        (yv, sv[0], zv[0])
    }

    fn assert_dql_eq(vals: Vec<f32>, dims: &[usize]) {
        let fused = run_fused(vals.clone(), dims);
        let expanded = run_expanded(vals, dims);
        assert_eq!(fused.0, expanded.0, "y mismatch");
        assert_eq!(fused.2, expanded.2, "zp mismatch");
        let ds = (fused.1 - expanded.1).abs();
        assert!(
            ds <= 1e-6,
            "scale mismatch fused={} expanded={}",
            fused.1,
            expanded.1
        );
    }

    #[test]
    fn official_onnx_dynamicquantizelinear() {
        // onnx `test_dynamicquantizelinear`. zp/scale match the published
        // vectors. `y` is compared to the expanded Burn formula (f32 `/ 255`),
        // which can sit 1 bin off numpy's f64-then-cast scale on exact-`.5`
        // ties (`-2.5`, `0.5`). That is the same ±1 int8 gap already seen vs
        // ort; fused must not add a second source of drift.
        let x = vec![0.0, 2.0, -3.0, -2.5, 1.34, 0.5];
        let (y, scale, zp) = run_fused(x.clone(), &[6]);
        assert_eq!(zp, 153);
        assert!((scale - 0.019607844).abs() < 1e-6, "scale={scale}");
        assert_eq!(y[0], 153);
        assert_eq!(y[1], 255);
        assert_eq!(y[2], 0);
        assert_eq!(y[4], 221);
        let expanded = run_expanded(x, &[6]);
        assert_eq!(y, expanded.0);
        assert_eq!(zp, expanded.2);
    }

    #[test]
    fn fused_matches_expanded_mixed_signs() {
        assert_dql_eq(vec![0.0, 2.0, -3.0, -2.5, 1.34, 0.5], &[6]);
    }

    #[test]
    fn fused_matches_expanded_all_positive() {
        // min adjusted to 0 → zp = 0
        assert_dql_eq(vec![0.1, 0.5, 1.0, 2.0, 3.5], &[5]);
    }

    #[test]
    fn fused_matches_expanded_all_negative() {
        assert_dql_eq(vec![-4.0, -1.0, -0.25, -8.0], &[4]);
    }

    #[test]
    fn fused_matches_expanded_rank3_attn_like() {
        let n = 2 * 4 * 8;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.07 - 3.2).collect();
        assert_dql_eq(x, &[2, 4, 8]);
    }

    #[test]
    fn fused_matches_expanded_e5_hidden() {
        // e5 hidden [S, 384] after LN, mixed signs around 0
        let n = 16 * 384;
        let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.13 - 1.1).collect();
        assert_dql_eq(x, &[16, 384]);
    }

    #[test]
    fn zero_tensor_is_all_zero() {
        let (y, scale, zp) = run_fused(vec![0.0; 8], &[8]);
        assert_eq!(zp, 0);
        assert_eq!(scale, 0.0);
        assert_eq!(y, vec![0u8; 8]);
    }

    #[test]
    fn ties_to_even_not_away_from_zero() {
        // Isolated kernel: scale=1, zp=0. `f32::round` is away-from-zero
        // (0.5 → 1); ONNX/Burn are ties-to-even (0.5 → 0, 1.5 → 2).
        assert_eq!(quant_one(0.5, 1.0, 0.0), 0);
        assert_eq!(quant_one(1.5, 1.0, 0.0), 2);
        assert_eq!(quant_one(-0.5, 1.0, 10.0), 10);
        assert_dql_eq(vec![0.5, 1.5, -0.5, 2.5], &[4]);
    }
}
