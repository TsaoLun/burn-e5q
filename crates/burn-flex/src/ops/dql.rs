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

fn avx512f_available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "std")))]
    {
        false
    }
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
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if avx512f_available() && x.len() >= 16 {
        // SAFETY: runtime `avx512f` check above.
        return unsafe { minmax_avx512(x) };
    }
    minmax_slice_scalar(x)
}

fn minmax_slice_scalar(x: &[f32]) -> (f32, f32) {
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

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn minmax_avx512(x: &[f32]) -> (f32, f32) {
    use core::arch::x86_64::*;
    unsafe {
        let mut vmin = _mm512_set1_ps(f32::INFINITY);
        let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
        let n = x.len();
        let mut i = 0;
        while i + 16 <= n {
            let v = _mm512_loadu_ps(x.as_ptr().add(i));
            vmin = _mm512_min_ps(vmin, v);
            vmax = _mm512_max_ps(vmax, v);
            i += 16;
        }
        let mut mn = _mm512_reduce_min_ps(vmin);
        let mut mx = _mm512_reduce_max_ps(vmax);
        while i < n {
            let v = *x.as_ptr().add(i);
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
            i += 1;
        }
        (mn, mx)
    }
}

fn quantize(x: &[f32], scale: f32, zp: f32) -> Vec<u8> {
    let mut y = vec![0u8; x.len()];
    #[cfg(feature = "rayon")]
    {
        if x.len() >= crate::ops::PARALLEL_THRESHOLD {
            y.par_chunks_mut(CHUNK)
                .zip(x.par_chunks(CHUNK))
                .for_each(|(dst, src)| quantize_slice(dst, src, scale, zp));
            return y;
        }
    }
    quantize_slice(&mut y, x, scale, zp);
    y
}

fn quantize_slice(dst: &mut [u8], src: &[f32], scale: f32, zp: f32) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if avx512f_available() && src.len() >= 16 {
        // SAFETY: runtime `avx512f` check above.
        unsafe { quantize_avx512(dst, src, scale, zp) };
        return;
    }
    for (o, &v) in dst.iter_mut().zip(src) {
        *o = quant_one(v, scale, zp);
    }
}

/// Same formula as [`quant_one`]: `round_ties_even(v / scale) + zp`, then clamp.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn quantize_avx512(dst: &mut [u8], src: &[f32], scale: f32, zp: f32) {
    use core::arch::x86_64::*;
    unsafe {
        let v_scale = _mm512_set1_ps(scale);
        let v_zp = _mm512_set1_ps(zp);
        let v_zero = _mm512_setzero_ps();
        let v_hi = _mm512_set1_ps(255.0);
        let n = src.len();
        let mut i = 0;
        while i + 16 <= n {
            let v = _mm512_loadu_ps(src.as_ptr().add(i));
            let q = _mm512_div_ps(v, v_scale);
            let r = _mm512_roundscale_ps(q, _MM_FROUND_TO_NEAREST_INT);
            let r = _mm512_add_ps(r, v_zp);
            let r = _mm512_min_ps(_mm512_max_ps(r, v_zero), v_hi);
            let bytes = _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(r));
            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), bytes);
            i += 16;
        }
        while i < n {
            *dst.get_unchecked_mut(i) = quant_one(*src.get_unchecked(i), scale, zp);
            i += 1;
        }
    }
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

    #[test]
    fn fused_matches_expanded_e5_512_hidden() {
        // Below PARALLEL_THRESHOLD (196K < 256K): serial SIMD body + tail.
        let n = 512 * 384;
        let x: Vec<f32> = (0..n).map(|i| ((i % 29) as f32) * 0.11 - 1.4).collect();
        assert_dql_eq(x, &[1, 512, 384]);
    }

    #[test]
    fn fused_matches_expanded_e5_512_ffn() {
        // Above PARALLEL_THRESHOLD: rayon chunks + SIMD.
        let n = 512 * 1536;
        let x: Vec<f32> = (0..n).map(|i| ((i % 47) as f32) * 0.09 - 2.1).collect();
        assert_dql_eq(x, &[1, 512, 1536]);
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn dql_avx512_faster_than_scalar_e5() {
        if !avx512f_available() {
            return;
        }
        let x: Vec<f32> = (0..512 * 1536)
            .map(|i| ((i % 47) as f32) * 0.09 - 2.1)
            .collect();
        let time_ms = |f: &mut dyn FnMut()| {
            let start = std::time::Instant::now();
            f();
            start.elapsed().as_secs_f64() * 1e3
        };
        let _ = dql_u8(&x);
        let mut simd_ms = f64::INFINITY;
        let mut scalar_ms = f64::INFINITY;
        for _ in 0..4 {
            simd_ms = simd_ms.min(time_ms(&mut || {
                let _ = dql_u8(&x);
            }));
            scalar_ms = scalar_ms.min(time_ms(&mut || {
                let (mn, mx) = minmax_slice_scalar(&x);
                let xmin = mn.min(0.0);
                let xmax = mx.max(0.0);
                let scale = (xmax - xmin) / 255.0;
                let zp = (0.0 - xmin / scale).round_ties_even().clamp(0.0, 255.0);
                let mut y = vec![0u8; x.len()];
                for (o, &v) in y.iter_mut().zip(&x) {
                    *o = quant_one(v, scale, zp);
                }
                std::hint::black_box(y);
            }));
        }
        println!(
            "e5 FFN [512,1536] DQL: avx512 {simd_ms:.2} ms, scalar {scalar_ms:.2} ms; ×12 ≈ {:.1} vs {:.1}",
            simd_ms * 12.0,
            scalar_ms * 12.0
        );
        assert!(
            simd_ms < scalar_ms * 0.55,
            "avx512 {simd_ms:.2} ms should beat scalar {scalar_ms:.2} ms"
        );
    }
}
