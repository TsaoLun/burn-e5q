//! AVX-512 GELU: `0.5 * x * (1 + erf(x / √2))` with the same piecewise
//! rationals as musl / libm `erff` (SunPro fdlibm `s_erff.c`).
//!
//! Not Abramowitz–Stegun 7.1.26 — that approximation was already rejected
//! because it drifts cosine. The default `erf` tensor op still calls
//! `libm::erff`; only the fused GELU sweep uses this kernel.
//!
//! ====================================================
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================

use crate::ops::unary::erf_f32;

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

#[inline]
pub(crate) fn gelu_one(v: f32, sqrt2: f32) -> f32 {
    0.5 * v * (1.0 + erf_f32(v / sqrt2))
}

/// In-place GELU. AVX-512 when present; scalar `libm::erff` otherwise.
pub(crate) fn gelu_inplace(data: &mut [f32], sqrt2: f32) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if available() {
        enable_ftz_daz();
        // SAFETY: runtime `avx512f` check above.
        unsafe { gelu_inplace_avx512(data, sqrt2) };
        return;
    }
    for x in data.iter_mut() {
        *x = gelu_one(*x, sqrt2);
    }
}

/// Out-of-place GELU. `dst` and `src` must have the same length and must
/// not alias (the in-place helper is [`gelu_inplace`]).
pub(crate) fn gelu_copy(dst: &mut [f32], src: &[f32], sqrt2: f32) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if available() {
        enable_ftz_daz();
        // SAFETY: runtime `avx512f` check above; `dst` / `src` do not alias.
        unsafe { gelu_copy_avx512(dst, src, sqrt2) };
        return;
    }
    for (o, &v) in dst.iter_mut().zip(src) {
        *o = gelu_one(v, sqrt2);
    }
}

/// Per-thread: flush denormals. Padded e5 rows otherwise spend a long time
/// in denormal `erff` / AVX-512 divides.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn enable_ftz_daz() {
    unsafe {
        let mut mxcsr: u32 = 0;
        core::arch::asm!(
            "stmxcsr [{ptr}]",
            "or dword ptr [{ptr}], {flags}",
            "ldmxcsr [{ptr}]",
            ptr = in(reg) &mut mxcsr,
            flags = const (1u32 << 15) | (1u32 << 6),
            options(nostack),
        );
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn gelu_inplace_avx512(data: &mut [f32], sqrt2: f32) {
    let n = data.len();
    let p = data.as_mut_ptr();
    unsafe { gelu_ptr_avx512(p, p, n, sqrt2) };
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn gelu_copy_avx512(dst: &mut [f32], src: &[f32], sqrt2: f32) {
    debug_assert_eq!(dst.len(), src.len());
    unsafe { gelu_ptr_avx512(dst.as_mut_ptr(), src.as_ptr(), dst.len(), sqrt2) };
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn gelu_ptr_avx512(dst: *mut f32, src: *const f32, n: usize, sqrt2: f32) {
    use core::arch::x86_64::*;
    unsafe {
        let v_sqrt2 = _mm512_set1_ps(sqrt2);
        let v_half = _mm512_set1_ps(0.5);
        let v_one = _mm512_set1_ps(1.0);
        let mut i = 0;
        while i + 16 <= n {
            let x = _mm512_loadu_ps(src.add(i));
            let t = _mm512_div_ps(x, v_sqrt2);
            let e = erf_ps_avx512(t);
            let y = _mm512_mul_ps(_mm512_mul_ps(v_half, x), _mm512_add_ps(v_one, e));
            _mm512_storeu_ps(dst.add(i), y);
            i += 16;
        }
        if i < n {
            let rem = n - i;
            let mask = ((1u32 << rem) - 1) as u16;
            let x = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, src.add(i));
            let t = _mm512_div_ps(x, v_sqrt2);
            let e = erf_ps_avx512(t);
            let y = _mm512_mul_ps(_mm512_mul_ps(v_half, x), _mm512_add_ps(v_one, e));
            _mm512_mask_storeu_ps(dst.add(i), mask, y);
        }
    }
}

// ---------------------------------------------------------------------------
// musl / fdlibm `erff` coefficients (hex comments from libm 0.2.16).
// ---------------------------------------------------------------------------

#[allow(clippy::excessive_precision)]
mod coef {
    pub const ERX: f32 = 8.4506291151e-01; /* 0x3f58560b */
    pub const PP0: f32 = 1.2837916613e-01; /* 0x3e0375d4 */
    pub const PP1: f32 = -3.2504209876e-01; /* 0xbea66beb */
    pub const PP2: f32 = -2.8481749818e-02; /* 0xbce9528f */
    pub const PP3: f32 = -5.7702702470e-03; /* 0xbbbd1489 */
    pub const PP4: f32 = -2.3763017452e-05; /* 0xb7c756b1 */
    pub const QQ1: f32 = 3.9791721106e-01; /* 0x3ecbbbce */
    pub const QQ2: f32 = 6.5022252500e-02; /* 0x3d852a63 */
    pub const QQ3: f32 = 5.0813062117e-03; /* 0x3ba68116 */
    pub const QQ4: f32 = 1.3249473704e-04; /* 0x390aee49 */
    pub const QQ5: f32 = -3.9602282413e-06; /* 0xb684e21a */
    pub const PA0: f32 = -2.3621185683e-03; /* 0xbb1acdc6 */
    pub const PA1: f32 = 4.1485610604e-01; /* 0x3ed46805 */
    pub const PA2: f32 = -3.7220788002e-01; /* 0xbebe9208 */
    pub const PA3: f32 = 3.1834661961e-01; /* 0x3ea2fe54 */
    pub const PA4: f32 = -1.1089469492e-01; /* 0xbde31cc2 */
    pub const PA5: f32 = 3.5478305072e-02; /* 0x3d1151b3 */
    pub const PA6: f32 = -2.1663755178e-03; /* 0xbb0df9c0 */
    pub const QA1: f32 = 1.0642088205e-01; /* 0x3dd9f331 */
    pub const QA2: f32 = 5.4039794207e-01; /* 0x3f0a5785 */
    pub const QA3: f32 = 7.1828655899e-02; /* 0x3d931ae7 */
    pub const QA4: f32 = 1.2617121637e-01; /* 0x3e013307 */
    pub const QA5: f32 = 1.3637083583e-02; /* 0x3c5f6e13 */
    pub const QA6: f32 = 1.1984500103e-02; /* 0x3c445aa3 */
    pub const RA0: f32 = -9.8649440333e-03; /* 0xbc21a093 */
    pub const RA1: f32 = -6.9385856390e-01; /* 0xbf31a0b7 */
    pub const RA2: f32 = -1.0558626175e+01; /* 0xc128f022 */
    pub const RA3: f32 = -6.2375331879e+01; /* 0xc2798057 */
    pub const RA4: f32 = -1.6239666748e+02; /* 0xc322658c */
    pub const RA5: f32 = -1.8460508728e+02; /* 0xc3389ae7 */
    pub const RA6: f32 = -8.1287437439e+01; /* 0xc2a2932b */
    pub const RA7: f32 = -9.8143291473e+00; /* 0xc11d077e */
    pub const SA1: f32 = 1.9651271820e+01; /* 0x419d35ce */
    pub const SA2: f32 = 1.3765776062e+02; /* 0x4309a863 */
    pub const SA3: f32 = 4.3456588745e+02; /* 0x43d9486f */
    pub const SA4: f32 = 6.4538726807e+02; /* 0x442158c9 */
    pub const SA5: f32 = 4.2900814819e+02; /* 0x43d6810b */
    pub const SA6: f32 = 1.0863500214e+02; /* 0x42d9451f */
    pub const SA7: f32 = 6.5702495575e+00; /* 0x40d23f7c */
    pub const SA8: f32 = -6.0424413532e-02; /* 0xbd777f97 */
    pub const RB0: f32 = -9.8649431020e-03; /* 0xbc21a092 */
    pub const RB1: f32 = -7.9928326607e-01; /* 0xbf4c9dd4 */
    pub const RB2: f32 = -1.7757955551e+01; /* 0xc18e104b */
    pub const RB3: f32 = -1.6063638306e+02; /* 0xc320a2ea */
    pub const RB4: f32 = -6.3756646729e+02; /* 0xc41f6441 */
    pub const RB5: f32 = -1.0250950928e+03; /* 0xc480230b */
    pub const RB6: f32 = -4.8351919556e+02; /* 0xc3f1c275 */
    pub const SB1: f32 = 3.0338060379e+01; /* 0x41f2b459 */
    pub const SB2: f32 = 3.2579251099e+02; /* 0x43a2e571 */
    pub const SB3: f32 = 1.5367296143e+03; /* 0x44c01759 */
    pub const SB4: f32 = 3.1998581543e+03; /* 0x4547fdbb */
    pub const SB5: f32 = 2.5530502930e+03; /* 0x451f90ce */
    pub const SB6: f32 = 4.7452853394e+02; /* 0x43ed43a7 */
    pub const SB7: f32 = -2.2440952301e+01; /* 0xc1b38712 */
    /// |x| < 0.84375
    pub const IX_SMALL: i32 = 0x3f58_0000;
    /// |x| < 1.25
    pub const IX_MID: i32 = 0x3fa0_0000;
    /// |x| < 1/0.35
    pub const IX_RA: i32 = 0x4036_db6d;
    /// |x| < 6
    pub const IX_SAT: i32 = 0x40c0_0000;
    pub const IX_INF: i32 = 0x7f80_0000;
}

/// 16-wide `erff`. Same regions as libm; the `|x| >= 1.25` tail uses two
/// Cephes `exp` evaluations in place of scalar `expf`.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
#[allow(unused_unsafe)]
unsafe fn erf_ps_avx512(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use coef::*;
    use core::arch::x86_64::*;
    unsafe {
        let sign_bit = _mm512_castsi512_ps(_mm512_set1_epi32(0x8000_0000u32 as i32));
        let abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fff_ffff));
        let ax = _mm512_and_ps(x, abs_mask);
        let ix = _mm512_castps_si512(ax);

        let mask_small = _mm512_cmplt_epu32_mask(ix, _mm512_set1_epi32(IX_SMALL));
        let mask_mid = _mm512_mask_cmplt_epu32_mask(
            _mm512_cmpge_epu32_mask(ix, _mm512_set1_epi32(IX_SMALL)),
            ix,
            _mm512_set1_epi32(IX_MID),
        );
        let mask_ra = _mm512_mask_cmplt_epu32_mask(
            _mm512_cmpge_epu32_mask(ix, _mm512_set1_epi32(IX_MID)),
            ix,
            _mm512_set1_epi32(IX_RA),
        );
        let mask_rb = _mm512_mask_cmplt_epu32_mask(
            _mm512_cmpge_epu32_mask(ix, _mm512_set1_epi32(IX_RA)),
            ix,
            _mm512_set1_epi32(IX_SAT),
        );
        let mask_sat = _mm512_mask_cmplt_epu32_mask(
            _mm512_cmpge_epu32_mask(ix, _mm512_set1_epi32(IX_SAT)),
            ix,
            _mm512_set1_epi32(IX_INF),
        );
        let mask_special = _mm512_cmpge_epu32_mask(ix, _mm512_set1_epi32(IX_INF));

        // |x| < 0.84375: x + x * P(z)/Q(z), z = x². Already signed.
        let z2 = _mm512_mul_ps(x, x);
        let mut rp = _mm512_set1_ps(PP4);
        rp = _mm512_fmadd_ps(rp, z2, _mm512_set1_ps(PP3));
        rp = _mm512_fmadd_ps(rp, z2, _mm512_set1_ps(PP2));
        rp = _mm512_fmadd_ps(rp, z2, _mm512_set1_ps(PP1));
        rp = _mm512_fmadd_ps(rp, z2, _mm512_set1_ps(PP0));
        let mut qq = _mm512_set1_ps(QQ5);
        qq = _mm512_fmadd_ps(qq, z2, _mm512_set1_ps(QQ4));
        qq = _mm512_fmadd_ps(qq, z2, _mm512_set1_ps(QQ3));
        qq = _mm512_fmadd_ps(qq, z2, _mm512_set1_ps(QQ2));
        qq = _mm512_fmadd_ps(qq, z2, _mm512_set1_ps(QQ1));
        qq = _mm512_fmadd_ps(qq, z2, _mm512_set1_ps(1.0));
        let y_small = _mm512_fmadd_ps(x, _mm512_div_ps(rp, qq), x);

        // 0.84375 <= |x| < 1.25: erf = ERX + P(s)/Q(s), s = |x| - 1.
        let s = _mm512_sub_ps(ax, _mm512_set1_ps(1.0));
        let mut pa = _mm512_set1_ps(PA6);
        pa = _mm512_fmadd_ps(pa, s, _mm512_set1_ps(PA5));
        pa = _mm512_fmadd_ps(pa, s, _mm512_set1_ps(PA4));
        pa = _mm512_fmadd_ps(pa, s, _mm512_set1_ps(PA3));
        pa = _mm512_fmadd_ps(pa, s, _mm512_set1_ps(PA2));
        pa = _mm512_fmadd_ps(pa, s, _mm512_set1_ps(PA1));
        pa = _mm512_fmadd_ps(pa, s, _mm512_set1_ps(PA0));
        let mut qa = _mm512_set1_ps(QA6);
        qa = _mm512_fmadd_ps(qa, s, _mm512_set1_ps(QA5));
        qa = _mm512_fmadd_ps(qa, s, _mm512_set1_ps(QA4));
        qa = _mm512_fmadd_ps(qa, s, _mm512_set1_ps(QA3));
        qa = _mm512_fmadd_ps(qa, s, _mm512_set1_ps(QA2));
        qa = _mm512_fmadd_ps(qa, s, _mm512_set1_ps(QA1));
        qa = _mm512_fmadd_ps(qa, s, _mm512_set1_ps(1.0));
        let y_mid = _mm512_add_ps(_mm512_set1_ps(ERX), _mm512_div_ps(pa, qa));

        let mut y = y_small;
        y = _mm512_mask_blend_ps(mask_mid, y, y_mid);

        let need_exp = mask_ra | mask_rb;
        if need_exp != 0 {
            // Dummy |x|=2 so 1/(x²) stays finite on lanes that skip this path.
            let ax_e = _mm512_mask_blend_ps(need_exp, _mm512_set1_ps(2.0), ax);
            let ss = _mm512_div_ps(_mm512_set1_ps(1.0), _mm512_mul_ps(ax_e, ax_e));

            let mut ra = _mm512_set1_ps(RA7);
            ra = _mm512_fmadd_ps(ra, ss, _mm512_set1_ps(RA6));
            ra = _mm512_fmadd_ps(ra, ss, _mm512_set1_ps(RA5));
            ra = _mm512_fmadd_ps(ra, ss, _mm512_set1_ps(RA4));
            ra = _mm512_fmadd_ps(ra, ss, _mm512_set1_ps(RA3));
            ra = _mm512_fmadd_ps(ra, ss, _mm512_set1_ps(RA2));
            ra = _mm512_fmadd_ps(ra, ss, _mm512_set1_ps(RA1));
            ra = _mm512_fmadd_ps(ra, ss, _mm512_set1_ps(RA0));
            let mut sa = _mm512_set1_ps(SA8);
            sa = _mm512_fmadd_ps(sa, ss, _mm512_set1_ps(SA7));
            sa = _mm512_fmadd_ps(sa, ss, _mm512_set1_ps(SA6));
            sa = _mm512_fmadd_ps(sa, ss, _mm512_set1_ps(SA5));
            sa = _mm512_fmadd_ps(sa, ss, _mm512_set1_ps(SA4));
            sa = _mm512_fmadd_ps(sa, ss, _mm512_set1_ps(SA3));
            sa = _mm512_fmadd_ps(sa, ss, _mm512_set1_ps(SA2));
            sa = _mm512_fmadd_ps(sa, ss, _mm512_set1_ps(SA1));
            sa = _mm512_fmadd_ps(sa, ss, _mm512_set1_ps(1.0));

            let mut rb = _mm512_set1_ps(RB6);
            rb = _mm512_fmadd_ps(rb, ss, _mm512_set1_ps(RB5));
            rb = _mm512_fmadd_ps(rb, ss, _mm512_set1_ps(RB4));
            rb = _mm512_fmadd_ps(rb, ss, _mm512_set1_ps(RB3));
            rb = _mm512_fmadd_ps(rb, ss, _mm512_set1_ps(RB2));
            rb = _mm512_fmadd_ps(rb, ss, _mm512_set1_ps(RB1));
            rb = _mm512_fmadd_ps(rb, ss, _mm512_set1_ps(RB0));
            let mut sb = _mm512_set1_ps(SB7);
            sb = _mm512_fmadd_ps(sb, ss, _mm512_set1_ps(SB6));
            sb = _mm512_fmadd_ps(sb, ss, _mm512_set1_ps(SB5));
            sb = _mm512_fmadd_ps(sb, ss, _mm512_set1_ps(SB4));
            sb = _mm512_fmadd_ps(sb, ss, _mm512_set1_ps(SB3));
            sb = _mm512_fmadd_ps(sb, ss, _mm512_set1_ps(SB2));
            sb = _mm512_fmadd_ps(sb, ss, _mm512_set1_ps(SB1));
            sb = _mm512_fmadd_ps(sb, ss, _mm512_set1_ps(1.0));

            let r = _mm512_mask_blend_ps(mask_rb, ra, rb);
            let big_s = _mm512_mask_blend_ps(mask_rb, sa, sb);

            let chopped = _mm512_and_si512(
                _mm512_castps_si512(ax_e),
                _mm512_set1_epi32(0xffff_e000u32 as i32),
            );
            let z = _mm512_castsi512_ps(chopped);
            let arg1 = _mm512_fnmadd_ps(z, z, _mm512_set1_ps(-0.5625));
            let corr = _mm512_fmadd_ps(
                _mm512_sub_ps(z, ax_e),
                _mm512_add_ps(z, ax_e),
                _mm512_div_ps(r, big_s),
            );
            let erfc = _mm512_div_ps(
                _mm512_mul_ps(exp_ps_avx512(arg1), exp_ps_avx512(corr)),
                ax_e,
            );
            let y_exp = _mm512_sub_ps(_mm512_set1_ps(1.0), erfc);
            y = _mm512_mask_blend_ps(need_exp, y, y_exp);
        }

        y = _mm512_mask_blend_ps(mask_sat, y, _mm512_set1_ps(1.0));
        // Apply sign on every region except the already-signed small poly.
        let y_signed = _mm512_or_ps(y, _mm512_and_ps(x, sign_bit));
        y = _mm512_mask_blend_ps(mask_small, y_signed, y_small);

        if mask_special != 0 {
            // libm: 1 - 2*sign + 1/x   (erf(±inf)=±1, erf(nan)=nan)
            let sign01 = _mm512_cvtepi32_ps(_mm512_srli_epi32(_mm512_castps_si512(x), 31));
            let special = _mm512_add_ps(
                _mm512_fnmadd_ps(sign01, _mm512_set1_ps(2.0), _mm512_set1_ps(1.0)),
                _mm512_div_ps(_mm512_set1_ps(1.0), x),
            );
            y = _mm512_mask_blend_ps(mask_special, y, special);
        }
        y
    }
}

/// Cephes `expf` on 16 lanes (same constants as the D=32 flash kernel).
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
#[allow(unused_unsafe)]
unsafe fn exp_ps_avx512(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    unsafe {
        let x = _mm512_min_ps(x, _mm512_set1_ps(88.376_26));
        let x = _mm512_max_ps(x, _mm512_set1_ps(-88.376_26));
        let fx = _mm512_mul_ps(x, _mm512_set1_ps(1.442_695));
        let n = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEAREST_INT);
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693_359_375), x);
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.121_944_40e-4), r);
        let mut y = _mm512_set1_ps(1.987_569_150_0e-4);
        y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.398_199_950_7e-3));
        y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.333_451_907_3e-3));
        y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.166_579_589_4e-2));
        y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.666_666_545_9e-1));
        y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.000_000_120_1e-1));
        let z = _mm512_mul_ps(r, r);
        y = _mm512_fmadd_ps(y, z, r);
        y = _mm512_add_ps(y, _mm512_set1_ps(1.0));
        _mm512_scalef_ps(y, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_slice_matches_libm_small_and_tail() {
        let sqrt2 = core::f32::consts::SQRT_2;
        let src: Vec<f32> = vec![-2.0, -0.5, 0.0, 0.5, 2.0, 3.5, -4.25, 1.1];
        let expected: Vec<f32> = src.iter().map(|&x| gelu_one(x, sqrt2)).collect();
        let mut out = vec![0.0f32; src.len()];
        gelu_copy(&mut out, &src, sqrt2);
        for (i, (a, b)) in out.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "gelu[{i}]: {a} vs libm {b} (err {:e})",
                (a - b).abs()
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn erf_avx512_tracks_libm_dense() {
        if !available() {
            return;
        }
        let mut xs = Vec::new();
        let mut i = -8000i32;
        while i <= 8000 {
            xs.push(i as f32 * 0.001);
            i += 1;
        }
        for &b in &[
            0.84375f32,
            0.8437499,
            0.8437501,
            1.25,
            1.249999,
            1.250001,
            f32::from_bits(0x4036_db6d),
            5.999,
            6.0,
            6.001,
            0.0,
            -0.0,
        ] {
            xs.push(b);
            xs.push(-b);
        }
        let n = xs.len();
        let mut got = vec![0.0f32; n];
        enable_ftz_daz();
        // SAFETY: test gated on avx512f; evaluate erf via GELU inversion
        // would lose the (1+erf) scale, so call the kernel through gelu
        // and also a raw erf helper.
        unsafe { erf_copy_avx512(&mut got, &xs) };

        let mut max_abs = 0.0f32;
        let mut worst = 0.0f32;
        let mut n_over = 0u32;
        for (&x, &g) in xs.iter().zip(&got) {
            if !x.is_finite() {
                continue;
            }
            let e = libm::erff(x);
            let err = (g - e).abs();
            if err > max_abs {
                max_abs = err;
                worst = x;
            }
            if err > 1.5e-6 {
                n_over += 1;
            }
            assert!(
                err < 3e-6,
                "erf({x}) simd={g} libm={e} err={err:e} (max {max_abs:e} at {worst})"
            );
        }
        println!("erf vs libm: {n} pts, max abs {max_abs:e} at {worst}, n>1.5e-6 = {n_over}");
        assert!(
            max_abs < 2e-6,
            "max abs error {max_abs:e} at {worst} exceeds 2e-6"
        );
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn gelu_e5_ffn_shape_matches_libm() {
        if !available() {
            return;
        }
        let sqrt2 = core::f32::consts::SQRT_2;
        let n = 512 * 1536;
        let src: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.041 - 1.85).collect();
        let mut out = vec![0.0f32; n];
        gelu_copy(&mut out, &src, sqrt2);
        let mut max_abs = 0.0f32;
        for (i, (&x, &g)) in src.iter().zip(&out).enumerate() {
            let e = gelu_one(x, sqrt2);
            let err = (g - e).abs();
            if err > max_abs {
                max_abs = err;
            }
            assert!(err < 2e-6, "gelu e5-shape[{i}] err {err:e}");
        }
        assert!(max_abs < 1.5e-6, "e5-shape max abs {max_abs:e}");
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn gelu_avx512_faster_than_scalar_e5_ffn() {
        if !available() {
            return;
        }
        let sqrt2 = core::f32::consts::SQRT_2;
        let n = 512 * 1536;
        let src: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.041 - 1.85).collect();

        let time_ms = |f: &mut dyn FnMut()| {
            let start = std::time::Instant::now();
            f();
            start.elapsed().as_secs_f64() * 1e3
        };

        let mut dst = vec![0.0f32; n];
        gelu_copy(&mut dst, &src, sqrt2);
        for x in dst.iter_mut() {
            *x = gelu_one(*x, sqrt2);
        }

        let mut simd_ms = f64::INFINITY;
        let mut scalar_ms = f64::INFINITY;
        for _ in 0..4 {
            simd_ms = simd_ms.min(time_ms(&mut || {
                gelu_copy(&mut dst, &src, sqrt2);
            }));
            scalar_ms = scalar_ms.min(time_ms(&mut || {
                for (o, &v) in dst.iter_mut().zip(&src) {
                    *o = gelu_one(v, sqrt2);
                }
            }));
        }
        println!(
            "e5 FFN [512,1536] GELU: avx512 {simd_ms:.2} ms, scalar libm {scalar_ms:.2} ms; ×12 ≈ {:.1} vs {:.1}",
            simd_ms * 12.0,
            scalar_ms * 12.0
        );
        assert!(
            simd_ms < scalar_ms * 0.45,
            "avx512 {simd_ms:.2} ms should beat scalar {scalar_ms:.2} ms by >2×"
        );
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn erf_copy_avx512(dst: &mut [f32], src: &[f32]) {
        use core::arch::x86_64::*;
        debug_assert_eq!(dst.len(), src.len());
        let n = dst.len();
        let mut i = 0;
        unsafe {
            while i + 16 <= n {
                let x = _mm512_loadu_ps(src.as_ptr().add(i));
                _mm512_storeu_ps(dst.as_mut_ptr().add(i), erf_ps_avx512(x));
                i += 16;
            }
            if i < n {
                let rem = n - i;
                let mask = ((1u32 << rem) - 1) as u16;
                let x = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, src.as_ptr().add(i));
                _mm512_mask_storeu_ps(dst.as_mut_ptr().add(i), mask, erf_ps_avx512(x));
            }
        }
    }
}
