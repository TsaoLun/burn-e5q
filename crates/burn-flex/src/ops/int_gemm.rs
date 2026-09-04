//! Blocked integer GEMM for flex (`u8`/`i8`/`i32` → `i32`).
//!
//! Row-major `C[m,n] += A[m,k] * B[k,n]` with i-k-j panels. The inner `j`
//! stream is contiguous on `C` and `B`. 8-bit inputs are widened **in the
//! inner product**, not as whole-tensor `int_cast`s.
//!
//! On x86_64+AVX512-VNNI the 8-bit path uses `vpdpbusd` (`u8 × i8 → i32`).
//! Other signedness combinations are mapped onto that instruction and
//! corrected with the same `sum_a` / `sum_b` used for zero-point fusion.
//!
//! Accumulators use wrapping i32 arithmetic to match ONNX MatMulInteger.

use alloc::vec;
use alloc::vec::Vec;
use core::any::TypeId;

/// K-panel: deep enough for e5 (`K=32/384/1536`) without blowing L1.
const KC: usize = 64;
/// M-panel / rayon grain.
const MC: usize = 64;
/// N-panel: inner vector stream (tiled fallback).
const NC: usize = 128;

/// Fan out over M once a single GEMM has enough MAC work.
/// `16×384×1536 ≈ 9e6` (e5 short FFN) stays serial — only one MC tile.
/// `512×384×384 ≈ 7.5e7` splits across cores.
const PARALLEL_OPS: usize = 8_000_000;

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[path = "int_gemm_vnni.rs"]
mod vnni;

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[path = "int_gemm_amx.rs"]
mod amx;

/// AVX-512-VNNI is present. Used by the integer-flash attention path.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
pub(crate) fn vnni_available() -> bool {
    vnni::available()
}

#[cfg(not(all(target_arch = "x86_64", feature = "std")))]
#[allow(dead_code)]
pub(crate) fn vnni_available() -> bool {
    false
}

/// `A[m,k] u8 @ B[k,n] u8 → i32[m,n]` into `out` (zeroed then filled).
pub(crate) fn gemm_u8u8_into(
    a: &[u8],
    b: &[u8],
    m: usize,
    n: usize,
    k: usize,
    zp: &Zp,
    out: &mut [i32],
) {
    debug_assert_eq!(a.len(), m.saturating_mul(k));
    debug_assert_eq!(b.len(), k.saturating_mul(n));
    debug_assert_eq!(out.len(), m.saturating_mul(n));
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if vnni::available() {
            vnni::gemm_into::<true, false>(a, b, m, n, k, zp, out);
            return;
        }
    }
    let tmp = gemm_with_zp(a, b, m, n, k, zp);
    out.copy_from_slice(&tmp);
}

pub(crate) trait AsAcc: Copy {
    fn as_acc(self) -> i32;
}

impl AsAcc for u8 {
    #[inline(always)]
    fn as_acc(self) -> i32 {
        self as i32
    }
}

impl AsAcc for i8 {
    #[inline(always)]
    fn as_acc(self) -> i32 {
        self as i32
    }
}

impl AsAcc for i32 {
    #[inline(always)]
    fn as_acc(self) -> i32 {
        self
    }
}

/// Per-row (`za`) / per-col (`zb`) zero-point, or a scalar broadcast.
#[derive(Clone, Debug)]
pub(crate) enum ZpLane {
    Zero,
    Scalar(i32),
    Per(Vec<i32>),
}

impl ZpLane {
    #[inline]
    pub(crate) fn at(&self, i: usize) -> i32 {
        match self {
            Self::Zero => 0,
            Self::Scalar(v) => *v,
            Self::Per(v) => v[i],
        }
    }

    #[inline]
    pub(crate) fn is_zero(&self) -> bool {
        match self {
            Self::Zero => true,
            Self::Scalar(v) => *v == 0,
            Self::Per(v) => v.iter().all(|&x| x == 0),
        }
    }
}

/// ONNX MatMulInteger zero-points for one `[M,K]×[K,N]` pair.
#[derive(Clone, Debug)]
pub(crate) struct Zp {
    pub za: ZpLane,
    pub zb: ZpLane,
}

impl Zp {
    pub(crate) const NONE: Self = Self {
        za: ZpLane::Zero,
        zb: ZpLane::Zero,
    };

    #[inline]
    pub(crate) fn is_none(&self) -> bool {
        self.za.is_zero() && self.zb.is_zero()
    }
}

/// `A[m,k] @ B[k,n] → i32[m,n]` (no zero-point).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn gemm<A, B>(a: &[A], b: &[B], m: usize, n: usize, k: usize) -> Vec<i32>
where
    A: AsAcc + Sync + Send + 'static + bytemuck::Pod,
    B: AsAcc + Sync + Send + 'static + bytemuck::Pod,
{
    gemm_with_zp(a, b, m, n, k, &Zp::NONE)
}

/// `A[m,k] @ B[k,n] → i32[m,n]`, then
/// `C -= za·sum_k(B) + sum_k(A)·zb − K·za·zb`.
pub(crate) fn gemm_with_zp<A, B>(
    a: &[A],
    b: &[B],
    m: usize,
    n: usize,
    k: usize,
    zp: &Zp,
) -> Vec<i32>
where
    A: AsAcc + Sync + Send + 'static + bytemuck::Pod,
    B: AsAcc + Sync + Send + 'static + bytemuck::Pod,
{
    debug_assert_eq!(a.len(), m.saturating_mul(k));
    debug_assert_eq!(b.len(), k.saturating_mul(n));

    if m == 0 || n == 0 {
        return vec![0i32; m.saturating_mul(n)];
    }
    if k == 0 {
        let mut c = vec![0i32; m.saturating_mul(n)];
        apply_zp(&mut c, m, n, 0, &[], &[], zp);
        return c;
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if amx::available() {
            if let Some(out) = try_amx(a, b, m, n, k, zp) {
                return out;
            }
        }
        if vnni::available() {
            if let Some(out) = try_vnni(a, b, m, n, k, zp) {
                return out;
            }
        }
    }

    gemm_tiled(a, b, m, n, k, zp)
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn try_vnni<A, B>(a: &[A], b: &[B], m: usize, n: usize, k: usize, zp: &Zp) -> Option<Vec<i32>>
where
    A: AsAcc + 'static + bytemuck::Pod,
    B: AsAcc + 'static + bytemuck::Pod,
{
    let ta = TypeId::of::<A>();
    let tb = TypeId::of::<B>();
    if ta == TypeId::of::<u8>() && tb == TypeId::of::<i8>() {
        return Some(vnni::gemm::<true, true>(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            m,
            n,
            k,
            zp,
        ));
    }
    if ta == TypeId::of::<u8>() && tb == TypeId::of::<u8>() {
        return Some(vnni::gemm::<true, false>(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            m,
            n,
            k,
            zp,
        ));
    }
    if ta == TypeId::of::<i8>() && tb == TypeId::of::<i8>() {
        return Some(vnni::gemm::<false, true>(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            m,
            n,
            k,
            zp,
        ));
    }
    if ta == TypeId::of::<i8>() && tb == TypeId::of::<u8>() {
        return Some(vnni::gemm::<false, false>(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            m,
            n,
            k,
            zp,
        ));
    }
    None
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn try_amx<A, B>(a: &[A], b: &[B], m: usize, n: usize, k: usize, zp: &Zp) -> Option<Vec<i32>>
where
    A: AsAcc + 'static + bytemuck::Pod,
    B: AsAcc + 'static + bytemuck::Pod,
{
    let ta = TypeId::of::<A>();
    let tb = TypeId::of::<B>();
    if ta == TypeId::of::<u8>() && tb == TypeId::of::<i8>() {
        return amx::gemm::<true, true>(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            m,
            n,
            k,
            zp,
        );
    }
    if ta == TypeId::of::<u8>() && tb == TypeId::of::<u8>() {
        return amx::gemm::<true, false>(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            m,
            n,
            k,
            zp,
        );
    }
    if ta == TypeId::of::<i8>() && tb == TypeId::of::<i8>() {
        return amx::gemm::<false, true>(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            m,
            n,
            k,
            zp,
        );
    }
    if ta == TypeId::of::<i8>() && tb == TypeId::of::<u8>() {
        return amx::gemm::<false, false>(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice(b),
            m,
            n,
            k,
            zp,
        );
    }
    None
}

fn gemm_tiled<A: AsAcc + Sync, B: AsAcc + Sync>(
    a: &[A],
    b: &[B],
    m: usize,
    n: usize,
    k: usize,
    zp: &Zp,
) -> Vec<i32> {
    let mut c = vec![0i32; m.saturating_mul(n)];

    #[cfg(feature = "rayon")]
    {
        let ops = m.saturating_mul(n).saturating_mul(k);
        if ops >= PARALLEL_OPS && m > MC {
            use rayon::prelude::*;
            let sum_a = needs_sum_a(zp).then(|| sum_rows(a, m, k));
            let sum_b = needs_sum_b(zp).then(|| sum_cols(b, k, n));
            c.par_chunks_mut(MC * n)
                .enumerate()
                .for_each(|(tile, c_tile)| {
                    let i0 = tile * MC;
                    let mb = c_tile.len() / n;
                    gemm_serial(&a[i0 * k..(i0 + mb) * k], b, c_tile, mb, n, k);
                    let za_tile = zp_rows_slice(&zp.za, i0, mb);
                    let zp_tile = Zp {
                        za: za_tile,
                        zb: zp.zb.clone(),
                    };
                    let sa = sum_a
                        .as_ref()
                        .map(|s| &s[i0..i0 + mb])
                        .unwrap_or(&[] as &[i32]);
                    let sb = sum_b.as_deref().unwrap_or(&[]);
                    apply_zp(c_tile, mb, n, k, sa, sb, &zp_tile);
                });
            return c;
        }
    }

    gemm_serial(a, b, &mut c, m, n, k);
    if !zp.is_none() {
        let sum_a = needs_sum_a(zp).then(|| sum_rows(a, m, k));
        let sum_b = needs_sum_b(zp).then(|| sum_cols(b, k, n));
        apply_zp(
            &mut c,
            m,
            n,
            k,
            sum_a.as_deref().unwrap_or(&[]),
            sum_b.as_deref().unwrap_or(&[]),
            zp,
        );
    }
    c
}

fn zp_rows_slice(za: &ZpLane, i0: usize, mb: usize) -> ZpLane {
    match za {
        ZpLane::Per(v) => ZpLane::Per(v[i0..i0 + mb].to_vec()),
        other => other.clone(),
    }
}

#[inline]
fn needs_sum_a(zp: &Zp) -> bool {
    !zp.zb.is_zero()
}

#[inline]
fn needs_sum_b(zp: &Zp) -> bool {
    !zp.za.is_zero()
}

fn sum_rows<A: AsAcc>(a: &[A], m: usize, k: usize) -> Vec<i32> {
    let mut s = vec![0i32; m];
    for i in 0..m {
        let mut acc = 0i32;
        for kk in 0..k {
            acc = acc.wrapping_add(a[i * k + kk].as_acc());
        }
        s[i] = acc;
    }
    s
}

fn sum_cols<B: AsAcc>(b: &[B], k: usize, n: usize) -> Vec<i32> {
    let mut s = vec![0i32; n];
    for kk in 0..k {
        let row = &b[kk * n..(kk + 1) * n];
        for (dst, &bv) in s.iter_mut().zip(row.iter()) {
            *dst = dst.wrapping_add(bv.as_acc());
        }
    }
    s
}

pub(crate) fn apply_zp(
    c: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
    sum_a: &[i32],
    sum_b: &[i32],
    zp: &Zp,
) {
    if zp.is_none() {
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if n.is_multiple_of(16) && std::is_x86_feature_detected!("avx512f") {
        // SAFETY: avx512f checked; `n` is 16-wide.
        unsafe {
            apply_zp_avx512(c, m, n, k, sum_a, sum_b, zp);
        }
        return;
    }
    apply_zp_scalar(c, m, n, k, sum_a, sum_b, zp);
}

fn apply_zp_scalar(
    c: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
    sum_a: &[i32],
    sum_b: &[i32],
    zp: &Zp,
) {
    let k_i = k as i32;
    for i in 0..m {
        let za = zp.za.at(i);
        let sa = if sum_a.is_empty() { 0 } else { sum_a[i] };
        let c_row = &mut c[i * n..i * n + n];
        for j in 0..n {
            let zb = zp.zb.at(j);
            let sb = if sum_b.is_empty() { 0 } else { sum_b[j] };
            let corr = za
                .wrapping_mul(sb)
                .wrapping_add(sa.wrapping_mul(zb))
                .wrapping_sub(k_i.wrapping_mul(za).wrapping_mul(zb));
            c_row[j] = c_row[j].wrapping_sub(corr);
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_zp_avx512(
    c: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
    sum_a: &[i32],
    sum_b: &[i32],
    zp: &Zp,
) {
    use core::arch::x86_64::*;
    let k_i = k as i32;
    let zb_scalar = match &zp.zb {
        ZpLane::Zero => Some(0),
        ZpLane::Scalar(v) => Some(*v),
        ZpLane::Per(_) => None,
    };
    unsafe {
        for i in 0..m {
            let za = zp.za.at(i);
            let sa = if sum_a.is_empty() { 0 } else { sum_a[i] };
            let vza = _mm512_set1_epi32(za);
            let vsa = _mm512_set1_epi32(sa);
            let vkza = _mm512_set1_epi32(k_i.wrapping_mul(za));
            let c_row = c.as_mut_ptr().add(i * n);
            let mut j = 0;
            while j < n {
                let sb = if sum_b.is_empty() {
                    _mm512_setzero_si512()
                } else {
                    _mm512_loadu_si512(sum_b.as_ptr().add(j).cast())
                };
                let zb = match zb_scalar {
                    Some(v) => _mm512_set1_epi32(v),
                    None => match &zp.zb {
                        ZpLane::Per(v) => _mm512_loadu_si512(v.as_ptr().add(j).cast()),
                        _ => _mm512_setzero_si512(),
                    },
                };
                // corr = za*sb + sa*zb - k*za*zb  (i32 wrap)
                let corr = _mm512_sub_epi32(
                    _mm512_add_epi32(_mm512_mullo_epi32(vza, sb), _mm512_mullo_epi32(vsa, zb)),
                    _mm512_mullo_epi32(vkza, zb),
                );
                let acc = _mm512_loadu_si512(c_row.add(j).cast());
                _mm512_storeu_si512(c_row.add(j).cast(), _mm512_sub_epi32(acc, corr));
                j += 16;
            }
        }
    }
}

fn gemm_serial<A: AsAcc, B: AsAcc>(a: &[A], b: &[B], c: &mut [i32], m: usize, n: usize, k: usize) {
    let mut k0 = 0;
    while k0 < k {
        let k1 = (k0 + KC).min(k);
        let mut i0 = 0;
        while i0 < m {
            let i1 = (i0 + MC).min(m);
            let mut j0 = 0;
            while j0 < n {
                let j1 = (j0 + NC).min(n);
                accum_panel(a, b, c, n, k, i0, i1, j0, j1, k0, k1);
                j0 = j1;
            }
            i0 = i1;
        }
        k0 = k1;
    }
}

#[inline]
fn accum_panel<A: AsAcc, B: AsAcc>(
    a: &[A],
    b: &[B],
    c: &mut [i32],
    n: usize,
    k: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    k0: usize,
    k1: usize,
) {
    let j_len = j1 - j0;
    for i in i0..i1 {
        let c_row = &mut c[i * n + j0..i * n + j1];
        let a_row = &a[i * k + k0..i * k + k1];
        for (kk, &aik) in a_row.iter().enumerate() {
            let av = aik.as_acc();
            let b_row = &b[(k0 + kk) * n + j0..(k0 + kk) * n + j0 + j_len];
            for (dst, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                *dst = dst.wrapping_add(av.wrapping_mul(bv.as_acc()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{gemm, gemm_with_zp, AsAcc, Zp, ZpLane};

    fn naive<A: AsAcc, B: AsAcc>(a: &[A], b: &[B], m: usize, n: usize, k: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for kk in 0..k {
                let av = a[i * k + kk].as_acc();
                for j in 0..n {
                    c[i * n + j] =
                        c[i * n + j].wrapping_add(av.wrapping_mul(b[kk * n + j].as_acc()));
                }
            }
        }
        c
    }

    fn naive_zp<A: AsAcc, B: AsAcc>(
        a: &[A],
        b: &[B],
        m: usize,
        n: usize,
        k: usize,
        zp: &Zp,
    ) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            let za = zp.za.at(i);
            for j in 0..n {
                let zb = zp.zb.at(j);
                let mut acc = 0i32;
                for kk in 0..k {
                    let av = a[i * k + kk].as_acc().wrapping_sub(za);
                    let bv = b[kk * n + j].as_acc().wrapping_sub(zb);
                    acc = acc.wrapping_add(av.wrapping_mul(bv));
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn gemm_u8_i8_matches_naive_ragged() {
        let m = 17;
        let n = 31;
        let k = 19;
        let a: Vec<u8> = (0..m * k).map(|i| (i * 7 + 3) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i * 5) as i16 - 80) as i8).collect();
        assert_eq!(gemm(&a, &b, m, n, k), naive(&a, &b, m, n, k));
    }

    #[test]
    fn gemm_i8_i8_matches_naive() {
        let m = 9;
        let n = 21;
        let k = 13;
        let a: Vec<i8> = (0..m * k).map(|i| ((i as i16 * 3) - 40) as i8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i as i16 * 5) - 90) as i8).collect();
        assert_eq!(gemm(&a, &b, m, n, k), naive(&a, &b, m, n, k));
    }

    #[test]
    fn gemm_u8_u8_matches_naive() {
        let m = 8;
        let n = 18;
        let k = 11;
        let a: Vec<u8> = (0..m * k).map(|i| (i * 9) as u8).collect();
        let b: Vec<u8> = (0..k * n).map(|i| (i * 3 + 1) as u8).collect();
        assert_eq!(gemm(&a, &b, m, n, k), naive(&a, &b, m, n, k));
    }

    #[test]
    fn gemm_i32_matches_naive_e5_like() {
        let m = 8;
        let n = 64;
        let k = 32;
        let a: Vec<i32> = (0..m * k).map(|i| (i as i32 % 17) - 8).collect();
        let b: Vec<i32> = (0..k * n).map(|i| (i as i32 % 13) - 6).collect();
        assert_eq!(gemm(&a, &b, m, n, k), naive(&a, &b, m, n, k));
    }

    #[test]
    fn gemm_k_zero_is_empty_product() {
        let a: [u8; 0] = [];
        let b: [i8; 0] = [];
        assert_eq!(gemm(&a, &b, 3, 4, 0), vec![0i32; 12]);
    }

    #[test]
    fn gemm_parallel_path_matches_naive() {
        // ops = 80*96*1280 ≈ 9.8e6 and m > MC, so the rayon M-split runs.
        let m = 80;
        let n = 96;
        let k = 1280;
        let a: Vec<u8> = (0..m * k).map(|i| i as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i as i16 - 64) as i8).collect();
        assert_eq!(gemm(&a, &b, m, n, k), naive(&a, &b, m, n, k));
    }

    #[test]
    fn gemm_scalar_zp_matches_centered_naive() {
        let m = 17;
        let n = 31;
        let k = 19;
        let a: Vec<u8> = (0..m * k).map(|i| (i * 7 + 3) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i * 5) as i16 - 80) as i8).collect();
        let zp = Zp {
            za: ZpLane::Scalar(17),
            zb: ZpLane::Scalar(-3),
        };
        assert_eq!(
            gemm_with_zp(&a, &b, m, n, k, &zp),
            naive_zp(&a, &b, m, n, k, &zp)
        );
    }

    #[test]
    fn gemm_per_col_zb_matches_centered_naive() {
        let m = 8;
        let n = 20;
        let k = 12;
        let a: Vec<u8> = (0..m * k).map(|i| (i * 11) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i as i16) - 40) as i8).collect();
        let zb: Vec<i32> = (0..n).map(|j| (j as i32 % 7) - 3).collect();
        let zp = Zp {
            za: ZpLane::Scalar(5),
            zb: ZpLane::Per(zb),
        };
        assert_eq!(
            gemm_with_zp(&a, &b, m, n, k, &zp),
            naive_zp(&a, &b, m, n, k, &zp)
        );
    }

    #[test]
    fn gemm_amx_aligned_u8_i8_matches_naive() {
        // 16×16×64 is one AMX tile; 64×64×384 is an e5-like QKV panel;
        // 32×1536×384 is a short FFN1 (same N/K as 512×384×1536, fewer rows).
        for (m, n, k) in [(16, 16, 64), (64, 64, 384), (32, 1536, 384)] {
            let a: Vec<u8> = (0..m * k).map(|i| (i * 13 + 9) as u8).collect();
            let b: Vec<i8> = (0..k * n).map(|i| ((i * 7) as i16 - 100) as i8).collect();
            let zp = Zp {
                za: ZpLane::Scalar(17),
                zb: ZpLane::Scalar(-3),
            };
            assert_eq!(
                gemm(&a, &b, m, n, k),
                naive(&a, &b, m, n, k),
                "no-zp {m}x{n}x{k}"
            );
            assert_eq!(
                gemm_with_zp(&a, &b, m, n, k, &zp),
                naive_zp(&a, &b, m, n, k, &zp),
                "zp {m}x{n}x{k}"
            );
        }
    }

    #[test]
    fn gemm_amx_ffn_pack_cache_beats_first_touch() {
        // Same i8 B pointer is reused (e5 weights). First call packs;
        // later calls should be compute-only.
        if super::try_amx::<u8, i8>(&[0u8; 16 * 64], &[0i8; 64 * 16], 16, 16, 64, &Zp::NONE)
            .is_none()
        {
            return;
        }
        let m = 512;
        let n = 1536;
        let k = 384;
        let a: Vec<u8> = (0..m * k).map(|i| (i * 13 + 9) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i * 7) as i16 - 100) as i8).collect();
        let b2 = b.clone();
        let zp = Zp {
            za: ZpLane::Scalar(17),
            zb: ZpLane::Scalar(-3),
        };
        let a32 = &a[..32 * k];
        assert_eq!(
            gemm_with_zp(a32, &b, 32, n, k, &zp),
            naive_zp(a32, &b, 32, n, k, &zp),
            "M=32 panel of the FFN1 B must match naive (pack+AMX)"
        );
        let t0 = std::time::Instant::now();
        let first = gemm_with_zp(&a, &b, m, n, k, &zp);
        let first_ms = t0.elapsed().as_secs_f64() * 1e3;
        let clone_t = std::time::Instant::now();
        let via_clone = gemm_with_zp(&a, &b2, m, n, k, &zp);
        let clone_ms = clone_t.elapsed().as_secs_f64() * 1e3;
        if via_clone != first {
            let i = via_clone
                .iter()
                .zip(&first)
                .position(|(x, y)| x != y)
                .unwrap();
            panic!(
                "two B allocs differ at {i} (row {} col {}): {} vs {} (AMX leftover, not cache)",
                i / n,
                i % n,
                via_clone[i],
                first[i]
            );
        }
        let mut later = f64::INFINITY;
        for _ in 0..4 {
            let t = std::time::Instant::now();
            let y = gemm_with_zp(&a, &b, m, n, k, &zp);
            later = later.min(t.elapsed().as_secs_f64() * 1e3);
            if y != first {
                let i = y
                    .iter()
                    .zip(&first)
                    .position(|(x, z)| x != z)
                    .expect("length mismatch");
                panic!(
                    "cached FFN1 mismatch at {i} (row {} col {}): {} vs {}; first {first_ms:.3} clone {clone_ms:.3}",
                    i / n,
                    i % n,
                    y[i],
                    first[i]
                );
            }
        }
        let _ = std::fs::write(
            "/tmp/amx_pack_bench.txt",
            format!("ffn1 512×384×1536 first {first_ms:.3} later {later:.3}\n"),
        );
        assert!(
            later < first_ms * 0.85 || later < 1.5,
            "cached FFN1 {later:.2} ms should beat first-touch {first_ms:.2} ms"
        );
    }

    #[test]
    fn gemm_e5_like_u8_i8_head() {
        // e5 attention head: M=seq, N=64? actually N=head*dim after reshape;
        // K=head_dim=32 is the small inner product.
        let m = 16;
        let n = 64;
        let k = 32;
        let a: Vec<u8> = (0..m * k).map(|i| (i * 13 + 9) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i * 7) as i16 - 100) as i8).collect();
        let zp = Zp {
            za: ZpLane::Scalar(128),
            zb: ZpLane::Scalar(0),
        };
        assert_eq!(
            gemm_with_zp(&a, &b, m, n, k, &zp),
            naive_zp(&a, &b, m, n, k, &zp)
        );
    }
}
