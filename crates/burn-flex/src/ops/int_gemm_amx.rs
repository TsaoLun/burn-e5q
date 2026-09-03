//! Intel AMX `tdpbusd` (`u8 × i8 → i32`) for large 8-bit GEMM.
//!
//! Rust's AMX intrinsics are still unstable (`x86_amx_intrinsics`). This
//! file uses the same instructions via `asm!`, which assembles on stable
//! LLVM. Tile geometry is the hardware max: 16×64 u8 × 64×16 i8 → 16×16 i32.
//!
//! e5 FFN / QKV (`512×384×1536`, `512×1536×384`, `512×384×384`) are exact
//! multiples of 16 / 16 / 64. Smaller or unaligned shapes return `None`
//! and fall back to VNNI.
//!
//! Linux may need `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`.

use super::{Zp, apply_zp};
use alloc::vec;
use alloc::vec::Vec;
use std::sync::OnceLock;

/// AMX tile: 16 rows × 64 bytes (u8 K=64 or i32 N=16).
const TM: usize = 16;
const TK: usize = 64;
const TN: usize = 16;

pub(super) fn available() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| cpu_has_amx_int8() && request_xtiledata())
}

fn cpu_has_amx_int8() -> bool {
    // CPUID.(EAX=7,ECX=0):EDX[24]=AMX-TILE, EDX[25]=AMX-INT8.
    #[cfg(target_arch = "x86_64")]
    {
        let cpuid = core::arch::x86_64::__cpuid_count(7, 0);
        (cpuid.edx & (1 << 24)) != 0 && (cpuid.edx & (1 << 25)) != 0
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_os = "linux")]
fn request_xtiledata() -> bool {
    // SYS_arch_prctl = 158; ARCH_REQ_XCOMP_PERM = 0x1023; XFEATURE_XTILEDATA = 18.
    let _ret: i64;
    unsafe {
        core::arch::asm!(
            "syscall",
            inout("rax") 158i64 => _ret,
            in("rdi") 0x1023i64,
            in("rsi") 18i64,
            out("rcx") _,
            out("r11") _,
            options(nostack, preserves_flags),
        );
    }
    probe_amx()
}

#[cfg(not(target_os = "linux"))]
fn request_xtiledata() -> bool {
    probe_amx()
}

fn probe_amx() -> bool {
    let mut cfg = [0u8; 64];
    cfg[0] = 1;
    for t in 0..3 {
        cfg[16 + t * 2] = 64;
        cfg[48 + t] = 16;
    }
    let a = [1u8; TM * TK];
    let b = [1u8; TM * TK];
    let mut c = [0i32; TM * TN];
    let sa = TK as i64;
    let sb = TK as i64;
    let sc = (TN * 4) as i64;
    unsafe {
        core::arch::asm!(
            "ldtilecfg [{cfg}]",
            "tilezero tmm0",
            "tileloadd tmm1, [{a} + {sa}]",
            "tileloadd tmm2, [{b} + {sb}]",
            "tdpbusd tmm0, tmm1, tmm2",
            "tilestored [{c} + {sc}], tmm0",
            "tilerelease",
            cfg = in(reg) cfg.as_ptr(),
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            c = in(reg) c.as_mut_ptr(),
            sa = in(reg) sa,
            sb = in(reg) sb,
            sc = in(reg) sc,
            options(nostack),
        );
    }
    c[0] == 64
}

/// `None` when the shape is not AMX-aligned (caller uses VNNI).
pub(super) fn gemm<const A_IS_U8: bool, const B_IS_I8: bool>(
    a: &[u8],
    b: &[u8],
    m: usize,
    n: usize,
    k: usize,
    zp: &Zp,
) -> Option<Vec<i32>> {
    if m == 0 || n == 0 {
        return Some(vec![0i32; m.saturating_mul(n)]);
    }
    if !m.is_multiple_of(TM) || !n.is_multiple_of(TN) || !k.is_multiple_of(TK) {
        return None;
    }
    if !available() {
        return None;
    }

    let da: i32 = if A_IS_U8 { 0 } else { 128 };
    let db: i32 = if B_IS_I8 { 0 } else { -128 };
    let a_work: Vec<u8>;
    let a_ref: &[u8] = if A_IS_U8 {
        a
    } else {
        a_work = a.iter().map(|x| x.wrapping_add(128)).collect();
        &a_work
    };
    let b_work: Vec<u8>;
    let b_ref: &[u8] = if B_IS_I8 {
        b
    } else {
        b_work = b.iter().map(|x| x.wrapping_sub(128)).collect();
        &b_work
    };

    let packed = pack_b_amx(b_ref, n, k);
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
                    gemm_tiles(a_ref, i0, mb, n, k, &packed, c_tile);
                });
            apply_amx_fixup(&mut c, a, b, m, n, k, da, db, zp);
            return Some(c);
        }
    }

    gemm_tiles(a_ref, 0, m, n, k, &packed, &mut c);
    apply_amx_fixup(&mut c, a, b, m, n, k, da, db, zp);
    Some(c)
}

fn apply_amx_fixup(
    c: &mut [i32],
    a: &[u8],
    b: &[u8],
    m: usize,
    n: usize,
    k: usize,
    da: i32,
    db: i32,
    zp: &Zp,
) {
    // tdpbusd computed (A+da) @ (B+db). Recover A@B, then ONNX zp.
    if da != 0 || db != 0 {
        let sum_a = sum_rows_orig(a, m, k, da == 128);
        let sum_b = sum_cols_orig(b, k, n, db == 0);
        let k_i = k as i32;
        let k_da_db = k_i.wrapping_mul(da).wrapping_mul(db);
        for i in 0..m {
            let sa = sum_a[i];
            for j in 0..n {
                let sb = sum_b[j];
                c[i * n + j] = c[i * n + j]
                    .wrapping_sub(da.wrapping_mul(sb))
                    .wrapping_sub(sa.wrapping_mul(db))
                    .wrapping_sub(k_da_db);
            }
        }
    }
    if !zp.is_none() {
        let signed_a = da == 128;
        let signed_b = db == 0;
        let sum_a = sum_rows_orig(a, m, k, signed_a);
        let sum_b = sum_cols_orig(b, k, n, signed_b);
        apply_zp(c, m, n, k, &sum_a, &sum_b, zp);
    }
}

fn sum_rows_orig(a: &[u8], m: usize, k: usize, signed: bool) -> Vec<i32> {
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

fn sum_cols_orig(b: &[u8], k: usize, n: usize, signed: bool) -> Vec<i32> {
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

/// B[k, n] → n-tiles of AMX B (16 rows × 64 bytes, K-groups of 4).
fn pack_b_amx(b: &[u8], n: usize, k: usize) -> Vec<u8> {
    let n_tiles = n / TN;
    let k_tiles = k / TK;
    let mut out = vec![0u8; n_tiles * k_tiles * TM * TK];
    for nt in 0..n_tiles {
        let n0 = nt * TN;
        for kt in 0..k_tiles {
            let k0 = kt * TK;
            let tile = &mut out[(nt * k_tiles + kt) * TM * TK..(nt * k_tiles + kt + 1) * TM * TK];
            for kg in 0..TM {
                for nj in 0..TN {
                    for t in 0..4 {
                        let kk = k0 + kg * 4 + t;
                        tile[kg * TK + nj * 4 + t] = b[kk * n + n0 + nj];
                    }
                }
            }
        }
    }
    out
}

fn gemm_tiles(
    a: &[u8],
    m0: usize,
    mb: usize,
    n: usize,
    k: usize,
    packed_b: &[u8],
    c: &mut [i32],
) {
    debug_assert!(mb.is_multiple_of(TM), "AMX path requires M multiple of 16");

    let mut cfg = [0u8; 64];
    cfg[0] = 1;
    for t in 0..3 {
        cfg[16 + t * 2] = 64;
        cfg[48 + t] = TM as u8;
    }

    let k_tiles = k / TK;
    let n_tiles = n / TN;
    let a_lda = k as i64;
    let b_ldb = TK as i64;
    let c_ldc = (n * 4) as i64;

    unsafe {
        core::arch::asm!(
            "ldtilecfg [{cfg}]",
            cfg = in(reg) cfg.as_ptr(),
            options(nostack),
        );
    }

    let mut row = 0;
    while row < mb {
        for nt in 0..n_tiles {
            unsafe {
                core::arch::asm!("tilezero tmm0", options(nomem, nostack));
            }
            for kt in 0..k_tiles {
                let a_ptr = unsafe { a.as_ptr().add((m0 + row) * k + kt * TK) };
                let b_ptr = unsafe { packed_b.as_ptr().add((nt * k_tiles + kt) * TM * TK) };
                unsafe {
                    core::arch::asm!(
                        "tileloadd tmm1, [{a} + {sa}]",
                        "tileloadd tmm2, [{b} + {sb}]",
                        "tdpbusd tmm0, tmm1, tmm2",
                        a = in(reg) a_ptr,
                        b = in(reg) b_ptr,
                        sa = in(reg) a_lda,
                        sb = in(reg) b_ldb,
                        options(nostack),
                    );
                }
            }
            let c_ptr = unsafe { c.as_mut_ptr().add(row * n + nt * TN) };
            unsafe {
                core::arch::asm!(
                    "tilestored [{c} + {sc}], tmm0",
                    c = in(reg) c_ptr,
                    sc = in(reg) c_ldc,
                    options(nostack),
                );
            }
        }
        row += TM;
    }

    unsafe {
        core::arch::asm!("tilerelease", options(nomem, nostack));
    }
}
