//! Affine dequant of an I32 GEMM product: `y = x.f32 * scale + bias`,
//! optionally followed by GELU in the same sweep.
//!
//! e5's FFN / QKV / out projections emit
//! `Cast(i32→f32) → Mul(scale) → Add(bias) [→ Gelu]`. Fusing that into one
//! pass avoids three extra walks of `[seq, 1536]` (or `[seq, 384]`).
//!
//! Fast path: contiguous I32 `x`, scalar F32 `scale`, last-axis (or scalar)
//! F32 `bias`. AVX-512 when present; large buffers also fan out with rayon.
//! Anything else falls back to `float + mul + add [+ gelu]`.

use burn_backend::{
    DType, FloatDType,
    ops::{ActivationOps, FloatTensorOps, IntTensorOps},
};
use burn_std::Bytes;

use crate::ops::gelu_avx512::gelu_one;
use crate::{Flex, FlexTensor, Layout};

/// `y = x.f32 * scale + bias`, optionally `gelu(y)`.
pub(crate) fn dequant_affine(
    value: FlexTensor,
    scale: FlexTensor,
    bias: FlexTensor,
    apply_gelu: bool,
) -> FlexTensor {
    if value.dtype() != DType::I32 || scale.dtype() != DType::F32 || bias.dtype() != DType::F32 {
        return fallback(value, scale, bias, apply_gelu);
    }

    let value = take_owned_contiguous(value);
    let scale = take_owned_contiguous(scale);
    let bias = take_owned_contiguous(bias);

    let n = value.layout().num_elements();
    let last = match value.layout().shape().last() {
        Some(&d) if d > 0 && n.is_multiple_of(d) => d,
        _ => return fallback(value, scale, bias, apply_gelu),
    };

    let scale_s = as_scalar_f32(&scale);
    let bias_row = as_last_axis_f32(&bias, last);
    let bias_s = as_scalar_f32(&bias);

    match (scale_s, bias_row, bias_s) {
        (Some(s), Some(b), _) => last_axis(value, s, b, last, apply_gelu),
        (Some(s), None, Some(b)) => scalar_bias(value, s, b, apply_gelu),
        _ => fallback(value, scale, bias, apply_gelu),
    }
}

fn fallback(
    value: FlexTensor,
    scale: FlexTensor,
    bias: FlexTensor,
    apply_gelu: bool,
) -> FlexTensor {
    let x = Flex::int_into_float(value, FloatDType::F32);
    let y = Flex::float_add(Flex::float_mul(x, scale), bias);
    if apply_gelu { Flex::gelu(y) } else { y }
}

fn take_owned_contiguous(tensor: FlexTensor) -> FlexTensor {
    if tensor.is_contiguous()
        && tensor.layout().start_offset() == 0
        && tensor.bytes().len()
            == tensor.layout().num_elements() * crate::tensor::dtype_size(tensor.dtype())
    {
        tensor
    } else {
        tensor.to_contiguous()
    }
}

fn as_scalar_f32(t: &FlexTensor) -> Option<f32> {
    if t.dtype() != DType::F32 || t.layout().num_elements() != 1 {
        return None;
    }
    Some(t.storage::<f32>()[0])
}

fn as_last_axis_f32(t: &FlexTensor, last: usize) -> Option<&[f32]> {
    if t.dtype() != DType::F32 || t.layout().num_elements() != last {
        return None;
    }
    if !t.is_contiguous() || t.layout().start_offset() != 0 {
        return None;
    }
    Some(&t.storage::<f32>()[..last])
}

fn last_axis(
    value: FlexTensor,
    scale: f32,
    bias: &[f32],
    last: usize,
    apply_gelu: bool,
) -> FlexTensor {
    let n = value.layout().num_elements();
    let shape = value.layout().shape().clone();
    let src: &[i32] = value.storage();
    let mut out = vec![0.0f32; n];
    apply_last_axis(&mut out, &src[..n], scale, bias, last, apply_gelu);
    FlexTensor::new(
        Bytes::from_elems(out),
        Layout::contiguous(shape),
        DType::F32,
    )
}

fn scalar_bias(value: FlexTensor, scale: f32, bias: f32, apply_gelu: bool) -> FlexTensor {
    let n = value.layout().num_elements();
    let shape = value.layout().shape().clone();
    let src: &[i32] = value.storage();
    let mut out = vec![0.0f32; n];
    apply_scalar(&mut out, &src[..n], scale, bias, apply_gelu);
    FlexTensor::new(
        Bytes::from_elems(out),
        Layout::contiguous(shape),
        DType::F32,
    )
}

fn apply_last_axis(
    dst: &mut [f32],
    src: &[i32],
    scale: f32,
    bias: &[f32],
    last: usize,
    apply_gelu: bool,
) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(src.len() % last, 0);
    debug_assert_eq!(bias.len(), last);

    #[cfg(feature = "rayon")]
    if src.len() >= crate::ops::PARALLEL_THRESHOLD {
        use rayon::prelude::*;
        dst.par_chunks_mut(last)
            .zip(src.par_chunks(last))
            .for_each(|(d, s)| row_last_axis(d, s, scale, bias, apply_gelu));
        return;
    }

    for (d, s) in dst.chunks_mut(last).zip(src.chunks(last)) {
        row_last_axis(d, s, scale, bias, apply_gelu);
    }
}

fn apply_scalar(dst: &mut [f32], src: &[i32], scale: f32, bias: f32, apply_gelu: bool) {
    debug_assert_eq!(dst.len(), src.len());

    #[cfg(feature = "rayon")]
    if src.len() >= crate::ops::PARALLEL_THRESHOLD {
        use rayon::prelude::*;
        const CHUNK: usize = 16 * 1024;
        dst.par_chunks_mut(CHUNK)
            .zip(src.par_chunks(CHUNK))
            .for_each(|(d, s)| row_scalar(d, s, scale, bias, apply_gelu));
        return;
    }

    row_scalar(dst, src, scale, bias, apply_gelu);
}

fn row_last_axis(dst: &mut [f32], src: &[i32], scale: f32, bias: &[f32], apply_gelu: bool) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(dst.len(), bias.len());

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if crate::ops::gelu_avx512::available() {
        crate::ops::gelu_avx512::enable_ftz_daz();
        // SAFETY: runtime `avx512f` check above.
        unsafe { row_last_axis_avx512(dst, src, scale, bias, apply_gelu) };
        return;
    }

    let sqrt2 = core::f32::consts::SQRT_2;
    for i in 0..dst.len() {
        let y = src[i] as f32 * scale + bias[i];
        dst[i] = if apply_gelu { gelu_one(y, sqrt2) } else { y };
    }
}

fn row_scalar(dst: &mut [f32], src: &[i32], scale: f32, bias: f32, apply_gelu: bool) {
    debug_assert_eq!(dst.len(), src.len());

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if crate::ops::gelu_avx512::available() {
        crate::ops::gelu_avx512::enable_ftz_daz();
        unsafe { row_scalar_avx512(dst, src, scale, bias, apply_gelu) };
        return;
    }

    let sqrt2 = core::f32::consts::SQRT_2;
    for i in 0..dst.len() {
        let y = src[i] as f32 * scale + bias;
        dst[i] = if apply_gelu { gelu_one(y, sqrt2) } else { y };
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn row_last_axis_avx512(
    dst: &mut [f32],
    src: &[i32],
    scale: f32,
    bias: &[f32],
    apply_gelu: bool,
) {
    use core::arch::x86_64::*;
    let n = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let bp = bias.as_ptr();
    unsafe {
        let vs = _mm512_set1_ps(scale);
        let mut i = 0;
        while i + 16 <= n {
            let vi = _mm512_loadu_si512(sp.add(i).cast());
            let vf = _mm512_cvtepi32_ps(vi);
            let mut y = _mm512_fmadd_ps(vf, vs, _mm512_loadu_ps(bp.add(i)));
            if apply_gelu {
                y = crate::ops::gelu_avx512::gelu_ps_avx512(y);
            }
            _mm512_storeu_ps(dp.add(i), y);
            i += 16;
        }
        if i < n {
            let rem = n - i;
            let mask = ((1u32 << rem) - 1) as u16;
            let vi = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, sp.add(i).cast());
            let vf = _mm512_cvtepi32_ps(vi);
            let mut y = _mm512_fmadd_ps(
                vf,
                vs,
                _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, bp.add(i)),
            );
            if apply_gelu {
                y = crate::ops::gelu_avx512::gelu_ps_avx512(y);
            }
            _mm512_mask_storeu_ps(dp.add(i), mask, y);
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn row_scalar_avx512(dst: &mut [f32], src: &[i32], scale: f32, bias: f32, apply_gelu: bool) {
    use core::arch::x86_64::*;
    let n = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    unsafe {
        let vs = _mm512_set1_ps(scale);
        let vb = _mm512_set1_ps(bias);
        let mut i = 0;
        while i + 16 <= n {
            let vi = _mm512_loadu_si512(sp.add(i).cast());
            let vf = _mm512_cvtepi32_ps(vi);
            let mut y = _mm512_fmadd_ps(vf, vs, vb);
            if apply_gelu {
                y = crate::ops::gelu_avx512::gelu_ps_avx512(y);
            }
            _mm512_storeu_ps(dp.add(i), y);
            i += 16;
        }
        if i < n {
            let rem = n - i;
            let mask = ((1u32 << rem) - 1) as u16;
            let vi = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, sp.add(i).cast());
            let vf = _mm512_cvtepi32_ps(vi);
            let mut y = _mm512_fmadd_ps(vf, vs, vb);
            if apply_gelu {
                y = crate::ops::gelu_avx512::gelu_ps_avx512(y);
            }
            _mm512_mask_storeu_ps(dp.add(i), mask, y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    fn flex_i32(data: Vec<i32>, shape: &[usize]) -> FlexTensor {
        FlexTensor::from_data(TensorData::new(data, shape.to_vec()))
    }

    fn flex_f32(data: Vec<f32>, shape: &[usize]) -> FlexTensor {
        FlexTensor::from_data(TensorData::new(data, shape.to_vec()))
    }

    fn ref_affine(src: &[i32], scale: f32, bias: &[f32], last: usize, gelu: bool) -> Vec<f32> {
        let sqrt2 = core::f32::consts::SQRT_2;
        src.iter()
            .enumerate()
            .map(|(i, &x)| {
                let y = x as f32 * scale + bias[i % last];
                if gelu { gelu_one(y, sqrt2) } else { y }
            })
            .collect()
    }

    fn assert_close(got: &[f32], exp: &[f32], tol: f32) {
        assert_eq!(got.len(), exp.len());
        let mut max = 0.0f32;
        for (i, (a, b)) in got.iter().zip(exp).enumerate() {
            let err = (a - b).abs();
            if err > max {
                max = err;
            }
            assert!(
                err < tol,
                "dequant_affine[{i}]: {a} vs {b} (err {err:e}, max {max:e})"
            );
        }
    }

    #[test]
    fn last_axis_matches_ref_small() {
        let src: Vec<i32> = vec![-4, -1, 0, 2, 7, -9, 3, 11];
        let bias = vec![0.1, -0.2, 0.3, 0.0];
        let scale = 0.017;
        let out = dequant_affine(
            flex_i32(src.clone(), &[2, 4]),
            flex_f32(vec![scale], &[]),
            flex_f32(bias.clone(), &[4]),
            false,
        );
        let got: &[f32] = out.storage();
        let exp = ref_affine(&src, scale, &bias, 4, false);
        assert_close(&got[..src.len()], &exp, 1e-6);
    }

    #[test]
    fn last_axis_gelu_matches_ref() {
        let src: Vec<i32> = (0..32).map(|i| (i % 13) as i32 - 6).collect();
        let bias: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 - 0.4).collect();
        let scale = 0.023;
        let out = dequant_affine(
            flex_i32(src.clone(), &[2, 16]),
            flex_f32(vec![scale], &[1]),
            flex_f32(bias.clone(), &[16]),
            true,
        );
        let got: &[f32] = out.storage();
        let exp = ref_affine(&src, scale, &bias, 16, true);
        assert_close(&got[..src.len()], &exp, 2e-6);
    }

    #[test]
    fn scalar_bias_and_tail() {
        let src: Vec<i32> = (0..19).map(|i| i as i32 - 8).collect();
        let scale = 0.5;
        let bias = -1.25;
        let out = dequant_affine(
            flex_i32(src.clone(), &[19]),
            flex_f32(vec![scale], &[1]),
            flex_f32(vec![bias], &[1]),
            false,
        );
        let got: &[f32] = out.storage();
        let exp = ref_affine(&src, scale, &[bias], 1, false);
        assert_close(&got[..src.len()], &exp, 1e-6);
    }

    #[test]
    fn e5_ffn1_shape_matches_ref() {
        let last = 1536;
        let rows = 32;
        let n = rows * last;
        let src: Vec<i32> = (0..n).map(|i| ((i % 97) as i32) - 48).collect();
        let bias: Vec<f32> = (0..last).map(|i| ((i % 31) as f32) * 0.01 - 0.15).collect();
        let scale = 0.0039;
        let out = dequant_affine(
            flex_i32(src.clone(), &[rows, last]),
            flex_f32(vec![scale], &[1, 1, 1]),
            flex_f32(bias.clone(), &[1, 1, last]),
            true,
        );
        let got: &[f32] = out.storage();
        let exp = ref_affine(&src, scale, &bias, last, true);
        assert_close(&got[..n], &exp, 2e-6);
    }

    #[test]
    fn unsqueezed_bias_is_last_axis() {
        let src = vec![1i32, 2, 3, 4, 5, 6];
        let bias = vec![10.0f32, 20.0, 30.0];
        let out = dequant_affine(
            flex_i32(src.clone(), &[2, 3]),
            flex_f32(vec![2.0], &[1, 1]),
            flex_f32(bias.clone(), &[1, 3]),
            false,
        );
        let got: &[f32] = out.storage();
        let exp = ref_affine(&src, 2.0, &bias, 3, false);
        assert_close(&got[..6], &exp, 1e-6);
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn fused_faster_than_cast_mul_add_gelu_e5_ffn() {
        if !crate::ops::gelu_avx512::available() {
            return;
        }
        let last = 1536;
        let rows = 512;
        let n = rows * last;
        let src: Vec<i32> = (0..n).map(|i| ((i % 97) as i32) - 48).collect();
        let bias: Vec<f32> = (0..last).map(|i| ((i % 31) as f32) * 0.01 - 0.15).collect();
        let scale = 0.0039;
        let x = flex_i32(src, &[rows, last]);
        let s = flex_f32(vec![scale], &[1]);
        let b = flex_f32(bias, &[last]);

        let time_ms = |f: &mut dyn FnMut()| {
            let start = std::time::Instant::now();
            f();
            start.elapsed().as_secs_f64() * 1e3
        };

        let mut fused = 0.0;
        let mut split = 0.0;
        for _ in 0..4 {
            fused += time_ms(&mut || {
                let _ = std::hint::black_box(dequant_affine(x.clone(), s.clone(), b.clone(), true));
            });
            split += time_ms(&mut || {
                let y = fallback(x.clone(), s.clone(), b.clone(), true);
                std::hint::black_box(y);
            });
        }
        fused /= 4.0;
        split /= 4.0;
        println!("dequant_affine+gelu [512,1536]: fused {fused:.3} ms, split {split:.3} ms");
        assert!(
            fused < split,
            "fused {fused:.3} ms should beat split {split:.3} ms"
        );
    }
}
