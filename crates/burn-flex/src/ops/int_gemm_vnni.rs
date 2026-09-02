//! AVX-512-VNNI `vpdpbusd` microkernel for 8-bit GEMM.
//!
//! `vpdpbusd` does 4× `u8 × i8 → i32` per 32-bit lane (16 lanes / ZMM).
//! Signedness that is not already `u8 × i8` is rewritten:
//!
//! - `i8` A is packed as `A + 128` (`da = 128`)
//! - `u8` B is packed as `B - 128` (`db = -128`)
//!
//! Combined with ONNX zero-points:
//! `C = C_vnni − (za+da)·sum_b − sum_a·(zb+db) − K·da·db + K·za·zb`.

use super::{Zp, ZpLane};
use alloc::vec;
use alloc::vec::Vec;

const VN: usize = 16;

pub(super) fn available() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
}

/// `A_IS_U8`: A is already unsigned. `B_IS_I8`: B is already signed.
pub(super) fn gemm<const A_IS_U8: bool, const B_IS_I8: bool>(
    a: &[u8],
    b: &[u8],
    m: usize,
    n: usize,
    k: usize,
    zp: &Zp,
) -> Vec<i32> {
    let da: i32 = if A_IS_U8 { 0 } else { 128 };
    let db: i32 = if B_IS_I8 { 0 } else { -128 };
    let need_sums = da != 0 || db != 0 || !zp.is_none();
    let sum_a = need_sums.then(|| sum_rows_u8(a, m, k, !A_IS_U8));
    let sum_b = need_sums.then(|| sum_cols_u8(b, k, n, B_IS_I8));

    let packed = pack_b::<B_IS_I8>(b, n, k);
    let mut c = vec![0i32; m * n];

    #[cfg(feature = "rayon")]
    {
        let ops = m.saturating_mul(n).saturating_mul(k);
        if ops >= super::PARALLEL_OPS && m > super::MC {
            use rayon::prelude::*;
            c.par_chunks_mut(super::MC * n)
                .enumerate()
                .for_each(|(tile, c_tile)| {
                    let i0 = tile * super::MC;
                    let mb = c_tile.len() / n;
                    let za_tile = match &zp.za {
                        ZpLane::Per(v) => ZpLane::Per(v[i0..i0 + mb].to_vec()),
                        other => other.clone(),
                    };
                    let sa = sum_a.as_ref().map(|s| &s[i0..i0 + mb]);
                    // SAFETY: `available()` checked avx512vnni.
                    unsafe {
                        gemm_rows::<A_IS_U8>(
                            &a[i0 * k..(i0 + mb) * k],
                            &packed,
                            c_tile,
                            mb,
                            n,
                            k,
                            da,
                            db,
                            &za_tile,
                            &zp.zb,
                            sa,
                            sum_b.as_deref(),
                        );
                    }
                });
            return c;
        }
    }

    // SAFETY: `available()` checked avx512vnni.
    unsafe {
        gemm_rows::<A_IS_U8>(
            a,
            &packed,
            &mut c,
            m,
            n,
            k,
            da,
            db,
            &zp.za,
            &zp.zb,
            sum_a.as_deref(),
            sum_b.as_deref(),
        );
    }
    c
}

/// Sum original (not VNNI-converted) values. `signed` means the bytes are `i8`.
fn sum_rows_u8(a: &[u8], m: usize, k: usize, signed: bool) -> Vec<i32> {
    let mut s = vec![0i32; m];
    for i in 0..m {
        let mut acc = 0i32;
        for kk in 0..k {
            acc = acc.wrapping_add(byte_as_acc(a[i * k + kk], signed));
        }
        s[i] = acc;
    }
    s
}

fn sum_cols_u8(b: &[u8], k: usize, n: usize, signed: bool) -> Vec<i32> {
    let mut s = vec![0i32; n];
    for kk in 0..k {
        for j in 0..n {
            s[j] = s[j].wrapping_add(byte_as_acc(b[kk * n + j], signed));
        }
    }
    s
}

#[inline]
fn byte_as_acc(b: u8, signed: bool) -> i32 {
    if signed {
        b as i8 as i32
    } else {
        b as i32
    }
}

fn pack_b<const B_IS_I8: bool>(b: &[u8], n: usize, k: usize) -> Vec<i32> {
    let n_tiles = n.div_ceil(VN);
    let k_groups = k.div_ceil(4);
    let mut out = vec![0i32; n_tiles * k_groups * VN];
    for nt in 0..n_tiles {
        let n0 = nt * VN;
        let n_len = (n - n0).min(VN);
        for kg in 0..k_groups {
            let k0 = kg * 4;
            for j in 0..n_len {
                let mut dword = 0u32;
                for t in 0..4 {
                    let kk = k0 + t;
                    let byte = if kk < k {
                        let raw = b[kk * n + n0 + j];
                        if B_IS_I8 {
                            raw
                        } else {
                            raw.wrapping_sub(128)
                        }
                    } else {
                        0
                    };
                    dword |= (byte as u32) << (8 * t);
                }
                out[nt * k_groups * VN + kg * VN + j] = dword as i32;
            }
        }
    }
    out
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn gemm_rows<const A_IS_U8: bool>(
    a: &[u8],
    packed_b: &[i32],
    c: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
    da: i32,
    db: i32,
    za: &ZpLane,
    zb: &ZpLane,
    sum_a: Option<&[i32]>,
    sum_b: Option<&[i32]>,
) {
    use core::arch::x86_64::*;

    let k_groups = k.div_ceil(4);
    let n_tiles = n.div_ceil(VN);
    let fixup = da != 0 || db != 0 || !za.is_zero() || !zb.is_zero();
    let k_i = k as i32;
    let k_da_db = k_i.wrapping_mul(da).wrapping_mul(db);

    for nt in 0..n_tiles {
        let n0 = nt * VN;
        let n_len = (n - n0).min(VN);
        let mask = if n_len == VN {
            0xffffu16
        } else {
            (1u16 << n_len) - 1
        };
        let b_base = nt * k_groups * VN;

        let sb = if let Some(sum_b) = sum_b {
            let mut tmp = [0i32; VN];
            tmp[..n_len].copy_from_slice(&sum_b[n0..n0 + n_len]);
            unsafe { _mm512_loadu_si512(tmp.as_ptr().cast()) }
        } else {
            _mm512_setzero_si512()
        };
        let zb_v = match zb {
            ZpLane::Zero => _mm512_setzero_si512(),
            ZpLane::Scalar(v) => _mm512_set1_epi32(*v),
            ZpLane::Per(v) => {
                let mut tmp = [0i32; VN];
                tmp[..n_len].copy_from_slice(&v[n0..n0 + n_len]);
                unsafe { _mm512_loadu_si512(tmp.as_ptr().cast()) }
            }
        };

        for i in 0..m {
            let mut acc = _mm512_setzero_si512();
            let a_row = &a[i * k..i * k + k];
            for kg in 0..k_groups {
                let k0 = kg * 4;
                let mut pack = 0u32;
                for t in 0..4 {
                    let kk = k0 + t;
                    let byte = if kk < k {
                        let raw = a_row[kk];
                        if A_IS_U8 {
                            raw
                        } else {
                            raw.wrapping_add(128)
                        }
                    } else {
                        0
                    };
                    pack |= (byte as u32) << (8 * t);
                }
                let a_bcast = _mm512_set1_epi32(pack as i32);
                let b_vec =
                    unsafe { _mm512_loadu_si512(packed_b.as_ptr().add(b_base + kg * VN).cast()) };
                acc = _mm512_dpbusd_epi32(acc, a_bcast, b_vec);
            }

            if fixup {
                let za_i = za.at(i);
                let sa = sum_a.map(|s| s[i]).unwrap_or(0);
                let za_da = _mm512_set1_epi32(za_i.wrapping_add(da));
                let sa_v = _mm512_set1_epi32(sa);
                let zb_db = _mm512_add_epi32(zb_v, _mm512_set1_epi32(db));
                // k*za*zb - k*da*db
                let k_term = _mm512_sub_epi32(
                    _mm512_mullo_epi32(_mm512_set1_epi32(k_i.wrapping_mul(za_i)), zb_v),
                    _mm512_set1_epi32(k_da_db),
                );
                acc = _mm512_add_epi32(
                    _mm512_sub_epi32(
                        _mm512_sub_epi32(acc, _mm512_mullo_epi32(za_da, sb)),
                        _mm512_mullo_epi32(sa_v, zb_db),
                    ),
                    k_term,
                );
            }

            unsafe {
                _mm512_mask_storeu_epi32(c.as_mut_ptr().add(i * n + n0), mask, acc);
            }
        }
    }
}
