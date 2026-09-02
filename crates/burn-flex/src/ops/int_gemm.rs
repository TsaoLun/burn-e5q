//! Blocked integer GEMM for flex (`u8`/`i8`/`i32` → `i32`).
//!
//! Row-major `C[m,n] += A[m,k] * B[k,n]` with i-k-j panels. The inner `j`
//! stream is contiguous on `C` and `B`, so LLVM can vectorize without a full
//! rhs transpose. 8-bit inputs are widened **in the inner product**, not as
//! whole-tensor `int_cast`s.
//!
//! Accumulators use wrapping i32 arithmetic to match the previous naive
//! kernel (ONNX MatMulInteger accumulates in int32).

use alloc::vec;
use alloc::vec::Vec;

/// K-panel: deep enough for e5 (`K=32/384/1536`) without blowing L1.
const KC: usize = 64;
/// M-panel / rayon grain.
const MC: usize = 64;
/// N-panel: inner vector stream.
const NC: usize = 128;

/// Fan out over M once a single GEMM has enough MAC work.
/// `16×384×1536 ≈ 9e6` (e5 short FFN) stays serial — only one MC tile.
/// `512×384×384 ≈ 7.5e7` splits across cores.
const PARALLEL_OPS: usize = 8_000_000;

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

/// `A[m,k] @ B[k,n] → i32[m,n]`.
pub(crate) fn gemm<A: AsAcc + Sync, B: AsAcc + Sync>(
    a: &[A],
    b: &[B],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<i32> {
    debug_assert_eq!(a.len(), m.saturating_mul(k));
    debug_assert_eq!(b.len(), k.saturating_mul(n));

    let mut c = vec![0i32; m.saturating_mul(n)];
    if m == 0 || n == 0 || k == 0 {
        return c;
    }

    #[cfg(feature = "rayon")]
    {
        let ops = m.saturating_mul(n).saturating_mul(k);
        if ops >= PARALLEL_OPS && m > MC {
            use rayon::prelude::*;
            c.par_chunks_mut(MC * n)
                .enumerate()
                .for_each(|(tile, c_tile)| {
                    let i0 = tile * MC;
                    let mb = c_tile.len() / n;
                    gemm_serial(&a[i0 * k..(i0 + mb) * k], b, c_tile, mb, n, k);
                });
            return c;
        }
    }

    gemm_serial(a, b, &mut c, m, n, k);
    c
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
            if av == 0 {
                continue;
            }
            let b_row = &b[(k0 + kk) * n + j0..(k0 + kk) * n + j0 + j_len];
            for (dst, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                *dst = dst.wrapping_add(av.wrapping_mul(bv.as_acc()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AsAcc, gemm};

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
}
