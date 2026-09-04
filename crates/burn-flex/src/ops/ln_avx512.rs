//! AVX-512 last-axis f32 LayerNorm for widths that are multiples of 16.
//!
//! Same biased variance as the macerator path:
//! `var = E[x²] − E[x]²`, then `y = (x − mean) * rsqrt(var + eps) * γ + β`.
//!
//! e5 uses D=384 (24 × 16) on every LN. Rows are blocked by 4 so γ/β stay
//! in registers across the normalize sweep. `dst == src` is allowed (unique
//! input written in place).

#[inline]
pub(crate) fn available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "std")))]
    {
        false
    }
}

/// Out-of-place. `dst` and `src` must have the same length.
pub(crate) fn rows_with_beta(
    src: &[f32],
    dst: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    d_model: usize,
    epsilon: f32,
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert!(d_model.is_multiple_of(16));
    debug_assert_eq!(src.len() % d_model, 0);
    debug_assert_eq!(gamma.len(), d_model);
    debug_assert_eq!(beta.len(), d_model);
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if available() {
        // SAFETY: avx512f checked; `d_model` is 16-wide.
        unsafe {
            ln_ptr_avx512::<true>(
                dst.as_mut_ptr(),
                src.as_ptr(),
                src.len(),
                gamma.as_ptr(),
                beta.as_ptr(),
                d_model,
                epsilon,
            );
        }
        return;
    }
    rows_scalar(src, dst, gamma, Some(beta), d_model, epsilon);
}

pub(crate) fn rows_no_beta(
    src: &[f32],
    dst: &mut [f32],
    gamma: &[f32],
    d_model: usize,
    epsilon: f32,
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert!(d_model.is_multiple_of(16));
    debug_assert_eq!(src.len() % d_model, 0);
    debug_assert_eq!(gamma.len(), d_model);
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if available() {
        unsafe {
            ln_ptr_avx512::<false>(
                dst.as_mut_ptr(),
                src.as_ptr(),
                src.len(),
                gamma.as_ptr(),
                core::ptr::null(),
                d_model,
                epsilon,
            );
        }
        return;
    }
    rows_scalar(src, dst, gamma, None, d_model, epsilon);
}

/// In-place. Two passes over each row: stats, then overwrite.
pub(crate) fn rows_with_beta_inplace(
    data: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    d_model: usize,
    epsilon: f32,
) {
    debug_assert!(d_model.is_multiple_of(16));
    debug_assert_eq!(data.len() % d_model, 0);
    debug_assert_eq!(gamma.len(), d_model);
    debug_assert_eq!(beta.len(), d_model);
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if available() {
        let p = data.as_mut_ptr();
        unsafe {
            ln_ptr_avx512::<true>(
                p,
                p,
                data.len(),
                gamma.as_ptr(),
                beta.as_ptr(),
                d_model,
                epsilon,
            );
        }
        return;
    }
    let tmp = data.to_vec();
    rows_scalar(&tmp, data, gamma, Some(beta), d_model, epsilon);
}

pub(crate) fn rows_no_beta_inplace(data: &mut [f32], gamma: &[f32], d_model: usize, epsilon: f32) {
    debug_assert!(d_model.is_multiple_of(16));
    debug_assert_eq!(data.len() % d_model, 0);
    debug_assert_eq!(gamma.len(), d_model);
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if available() {
        let p = data.as_mut_ptr();
        unsafe {
            ln_ptr_avx512::<false>(
                p,
                p,
                data.len(),
                gamma.as_ptr(),
                core::ptr::null(),
                d_model,
                epsilon,
            );
        }
        return;
    }
    let tmp = data.to_vec();
    rows_scalar(&tmp, data, gamma, None, d_model, epsilon);
}

fn rows_scalar(
    src: &[f32],
    dst: &mut [f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    d_model: usize,
    epsilon: f32,
) {
    for (in_row, out_row) in src.chunks(d_model).zip(dst.chunks_mut(d_model)) {
        let n = d_model as f32;
        let mut sum = 0.0f32;
        let mut sumsq = 0.0f32;
        for &x in in_row {
            sum += x;
            sumsq += x * x;
        }
        let mean = sum / n;
        let var = (sumsq / n) - mean * mean;
        let inv = 1.0 / (var + epsilon).sqrt();
        for i in 0..d_model {
            let y = (in_row[i] - mean) * inv * gamma[i];
            out_row[i] = match beta {
                Some(b) => y + b[i],
                None => y,
            };
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn ln_ptr_avx512<const HAS_BETA: bool>(
    dst: *mut f32,
    src: *const f32,
    n: usize,
    gamma: *const f32,
    beta: *const f32,
    d_model: usize,
    epsilon: f32,
) {
    debug_assert!(d_model.is_multiple_of(16));
    debug_assert_eq!(n % d_model, 0);
    let n_rows = n / d_model;
    let n_f = d_model as f32;
    unsafe {
        let mut row = 0usize;
        while row + 4 <= n_rows {
            let s0 = src.add(row * d_model);
            let s1 = src.add((row + 1) * d_model);
            let s2 = src.add((row + 2) * d_model);
            let s3 = src.add((row + 3) * d_model);
            let (m0, i0) = stats_row(s0, d_model, n_f, epsilon);
            let (m1, i1) = stats_row(s1, d_model, n_f, epsilon);
            let (m2, i2) = stats_row(s2, d_model, n_f, epsilon);
            let (m3, i3) = stats_row(s3, d_model, n_f, epsilon);
            affine_4::<HAS_BETA>(
                dst.add(row * d_model),
                dst.add((row + 1) * d_model),
                dst.add((row + 2) * d_model),
                dst.add((row + 3) * d_model),
                s0,
                s1,
                s2,
                s3,
                gamma,
                beta,
                d_model,
                m0,
                i0,
                m1,
                i1,
                m2,
                i2,
                m3,
                i3,
            );
            row += 4;
        }
        while row < n_rows {
            let s = src.add(row * d_model);
            let (mean, inv) = stats_row(s, d_model, n_f, epsilon);
            affine_1::<HAS_BETA>(dst.add(row * d_model), s, gamma, beta, d_model, mean, inv);
            row += 1;
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn stats_row(
    src: *const f32,
    d_model: usize,
    n_f: f32,
    epsilon: f32,
) -> (core::arch::x86_64::__m512, core::arch::x86_64::__m512) {
    use core::arch::x86_64::*;
    unsafe {
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut sq0 = _mm512_setzero_ps();
        let mut sq1 = _mm512_setzero_ps();
        let mut j = 0usize;
        while j + 32 <= d_model {
            let a = _mm512_loadu_ps(src.add(j));
            let b = _mm512_loadu_ps(src.add(j + 16));
            acc0 = _mm512_add_ps(acc0, a);
            acc1 = _mm512_add_ps(acc1, b);
            sq0 = _mm512_fmadd_ps(a, a, sq0);
            sq1 = _mm512_fmadd_ps(b, b, sq1);
            j += 32;
        }
        while j < d_model {
            let a = _mm512_loadu_ps(src.add(j));
            acc0 = _mm512_add_ps(acc0, a);
            sq0 = _mm512_fmadd_ps(a, a, sq0);
            j += 16;
        }
        let sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
        let sumsq = _mm512_reduce_add_ps(_mm512_add_ps(sq0, sq1));
        let mean = sum / n_f;
        let var = (sumsq / n_f) - mean * mean;
        let inv = 1.0 / (var + epsilon).sqrt();
        (_mm512_set1_ps(mean), _mm512_set1_ps(inv))
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn affine_1<const HAS_BETA: bool>(
    dst: *mut f32,
    src: *const f32,
    gamma: *const f32,
    beta: *const f32,
    d_model: usize,
    mean: core::arch::x86_64::__m512,
    inv: core::arch::x86_64::__m512,
) {
    use core::arch::x86_64::*;
    unsafe {
        let mut j = 0usize;
        while j < d_model {
            let x = _mm512_loadu_ps(src.add(j));
            let g = _mm512_loadu_ps(gamma.add(j));
            let y = _mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(x, mean), inv), g);
            let y = if HAS_BETA {
                _mm512_add_ps(y, _mm512_loadu_ps(beta.add(j)))
            } else {
                y
            };
            _mm512_storeu_ps(dst.add(j), y);
            j += 16;
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn affine_4<const HAS_BETA: bool>(
    d0: *mut f32,
    d1: *mut f32,
    d2: *mut f32,
    d3: *mut f32,
    s0: *const f32,
    s1: *const f32,
    s2: *const f32,
    s3: *const f32,
    gamma: *const f32,
    beta: *const f32,
    d_model: usize,
    m0: core::arch::x86_64::__m512,
    i0: core::arch::x86_64::__m512,
    m1: core::arch::x86_64::__m512,
    i1: core::arch::x86_64::__m512,
    m2: core::arch::x86_64::__m512,
    i2: core::arch::x86_64::__m512,
    m3: core::arch::x86_64::__m512,
    i3: core::arch::x86_64::__m512,
) {
    use core::arch::x86_64::*;
    unsafe {
        let mut j = 0usize;
        while j < d_model {
            let g = _mm512_loadu_ps(gamma.add(j));
            let b = if HAS_BETA {
                _mm512_loadu_ps(beta.add(j))
            } else {
                _mm512_setzero_ps()
            };
            let apply = |dst: *mut f32, src: *const f32, mean, inv| {
                let x = _mm512_loadu_ps(src.add(j));
                let y = _mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(x, mean), inv), g);
                let y = if HAS_BETA { _mm512_add_ps(y, b) } else { y };
                _mm512_storeu_ps(dst.add(j), y);
            };
            apply(d0, s0, m0, i0);
            apply(d1, s1, m1, i1);
            apply(d2, s2, m2, i2);
            apply(d3, s3, m3, i3);
            j += 16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_row(src: &[f32], gamma: &[f32], beta: Option<&[f32]>, eps: f32) -> Vec<f32> {
        let n = src.len() as f32;
        let sum: f32 = src.iter().sum();
        let sumsq: f32 = src.iter().map(|x| x * x).sum();
        let mean = sum / n;
        let var = (sumsq / n) - mean * mean;
        let inv = 1.0 / (var + eps).sqrt();
        src.iter()
            .enumerate()
            .map(|(i, &x)| {
                let y = (x - mean) * inv * gamma[i];
                match beta {
                    Some(b) => y + b[i],
                    None => y,
                }
            })
            .collect()
    }

    #[test]
    fn ln_e5_hidden_matches_sumsq_ref() {
        let rows = 32;
        let d = 384;
        let src: Vec<f32> = (0..rows * d)
            .map(|i| ((i % 97) as f32) * 0.031 - 1.4)
            .collect();
        let gamma: Vec<f32> = (0..d).map(|i| 0.7 + (i as f32) * 0.001).collect();
        let beta: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.002) - 0.3).collect();
        let mut out = vec![0.0f32; src.len()];
        rows_with_beta(&src, &mut out, &gamma, &beta, d, 1e-12);
        let mut max_abs = 0.0f32;
        for r in 0..rows {
            let a = &src[r * d..(r + 1) * d];
            let got = &out[r * d..(r + 1) * d];
            let exp = ref_row(a, &gamma, Some(&beta), 1e-12);
            for (g, e) in got.iter().zip(&exp) {
                let err = (g - e).abs();
                if err > max_abs {
                    max_abs = err;
                }
            }
        }
        assert!(max_abs < 2e-5, "D=384 max abs vs sumsq-ref {max_abs:e}");
    }

    #[test]
    fn ln_inplace_matches_copy() {
        let d = 384;
        let rows = 8;
        let src: Vec<f32> = (0..rows * d)
            .map(|i| ((i % 53) as f32) * 0.02 - 0.5)
            .collect();
        let gamma = vec![1.1f32; d];
        let beta = vec![-0.2f32; d];
        let mut copy = vec![0.0f32; src.len()];
        rows_with_beta(&src, &mut copy, &gamma, &beta, d, 1e-5);
        let mut inplace = src.clone();
        rows_with_beta_inplace(&mut inplace, &gamma, &beta, d, 1e-5);
        for (a, b) in copy.iter().zip(&inplace) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn ln_avx512_faster_than_scalar_e5_25x() {
        if !available() {
            return;
        }
        let rows = 512;
        let d = 384;
        let src: Vec<f32> = (0..rows * d)
            .map(|i| ((i % 97) as f32) * 0.031 - 1.4)
            .collect();
        let gamma: Vec<f32> = (0..d).map(|i| 0.8 + (i as f32) * 0.0005).collect();
        let beta: Vec<f32> = (0..d).map(|i| (i as f32) * 0.001 - 0.2).collect();
        let mut dst = vec![0.0f32; src.len()];
        rows_with_beta(&src, &mut dst, &gamma, &beta, d, 1e-12);

        let time_ms = |f: &mut dyn FnMut()| {
            let t = std::time::Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        };
        let mut simd_ms = f64::INFINITY;
        let mut scalar_ms = f64::INFINITY;
        for _ in 0..5 {
            simd_ms = simd_ms.min(time_ms(&mut || {
                rows_with_beta(&src, &mut dst, &gamma, &beta, d, 1e-12);
            }));
            scalar_ms = scalar_ms.min(time_ms(&mut || {
                rows_scalar(&src, &mut dst, &gamma, Some(&beta), d, 1e-12);
            }));
        }
        let _ = std::fs::write(
            "/tmp/ln_avx512_bench.txt",
            format!(
                "LN [512,384] avx512 {simd_ms:.3} scalar {scalar_ms:.3}; ×25 ≈ {:.1} vs {:.1}\n",
                simd_ms * 25.0,
                scalar_ms * 25.0
            ),
        );
        assert!(
            simd_ms < scalar_ms * 0.55 || simd_ms < 0.35,
            "avx512 {simd_ms:.3} ms should beat scalar {scalar_ms:.3} ms"
        );
    }
}
