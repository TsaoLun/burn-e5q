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

use super::{apply_zp, Zp};
use alloc::vec;
use alloc::vec::Vec;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

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

    // i8 weights (e5) keep the caller's pointer — cache the AMX B layout
    // and column sums across forwards. Converted u8 B is a temporary and
    // is packed without caching.
    let packed = if B_IS_I8 {
        cached_pack_b(b, n, k)
    } else {
        Arc::new(pack_b_amx(b_ref, n, k))
    };
    let sum_b = if B_IS_I8 {
        cached_sum_cols(b, k, n, true)
    } else {
        Arc::new(sum_cols_orig(b, k, n, false))
    };
    let mut c = vec![0i32; m * n];
    // AMX tile XSTATE is per logical CPU. Rayon workers on sibling
    // hyperthreads (or a `nomem` tilezero reordered across tilestored)
    // were observed to corrupt exactly one 16×16 C tile. Serial compute
    // is enough: after packed-B is cached, a 512×384×1536 is ~1 ms.
    gemm_tiles(a_ref, 0, m, n, k, &packed, &mut c);
    apply_amx_fixup(&mut c, a, m, n, k, da, db, zp, Some(sum_b.as_slice()));
    Some(c)
}

fn apply_amx_fixup(
    c: &mut [i32],
    a: &[u8],
    m: usize,
    n: usize,
    k: usize,
    da: i32,
    db: i32,
    zp: &Zp,
    sum_b: Option<&[i32]>,
) {
    // tdpbusd computed (A+da) @ (B+db). Recover A@B, then ONNX zp.
    // Column sums of B are cached with the packed weights; row sums of A
    // change every token and are always recomputed.
    let need_fixup = da != 0 || db != 0;
    let need_zp = !zp.is_none();
    if !need_fixup && !need_zp {
        return;
    }
    let signed_a = da == 128;
    let sum_a = sum_rows_orig(a, m, k, signed_a);
    let sum_b_owned;
    let sum_b: &[i32] = match sum_b {
        Some(s) => s,
        None => {
            sum_b_owned = vec![0i32; n];
            &sum_b_owned
        }
    };
    if need_fixup {
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
    if need_zp {
        apply_zp(c, m, n, k, &sum_a, sum_b, zp);
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
    #[cfg(target_arch = "x86_64")]
    if n.is_multiple_of(16) && std::is_x86_feature_detected!("avx512f") {
        // SAFETY: avx512f checked; `n` is 16-wide.
        return unsafe { sum_cols_avx512(b, k, n, signed) };
    }
    let mut s = vec![0i32; n];
    for kk in 0..k {
        for j in 0..n {
            s[j] = s[j].wrapping_add(byte_as_acc(b[kk * n + j], signed));
        }
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_cols_avx512(b: &[u8], k: usize, n: usize, signed: bool) -> Vec<i32> {
    use core::arch::x86_64::*;
    let mut s = vec![0i32; n];
    unsafe {
        for kk in 0..k {
            let row = b.as_ptr().add(kk * n);
            let mut j = 0;
            while j < n {
                let bytes = _mm_loadu_si128(row.add(j).cast());
                let v = if signed {
                    _mm512_cvtepi8_epi32(bytes)
                } else {
                    _mm512_cvtepu8_epi32(bytes)
                };
                let acc = _mm512_loadu_si512(s.as_ptr().add(j).cast());
                _mm512_storeu_si512(s.as_mut_ptr().add(j).cast(), _mm512_add_epi32(acc, v));
                j += 16;
            }
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

struct PackCache {
    packs: HashMap<(usize, usize, usize, u64), Arc<Vec<u8>>>,
    sums: HashMap<(usize, usize, usize, bool, u64), Arc<Vec<i32>>>,
}

fn amx_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn pack_cache() -> &'static Mutex<PackCache> {
    static CACHE: OnceLock<Mutex<PackCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(PackCache {
            packs: HashMap::new(),
            sums: HashMap::new(),
        })
    })
}

/// Cheap fingerprint so a freed-then-reused pointer with different bytes
/// does not return the previous packed tile (tests reuse same-size allocs).
fn buf_tag(b: &[u8]) -> u64 {
    let n = b.len() as u64;
    let mut t = n;
    if b.len() >= 8 {
        t ^= u64::from_le_bytes(b[..8].try_into().unwrap());
        t ^= u64::from_le_bytes(b[b.len() - 8..].try_into().unwrap());
    }
    if b.len() >= 16 {
        let mid = b.len() / 2;
        t ^= u64::from_le_bytes(b[mid - 4..mid + 4].try_into().unwrap());
    }
    t
}

/// Cache packed AMX-B by the weight buffer pointer. e5 i8 weights live for
/// the process; a freed-then-reused pointer is evicted when the map hits 96
/// or when `buf_tag` disagrees.
fn cached_pack_b(b: &[u8], n: usize, k: usize) -> Arc<Vec<u8>> {
    let key = (b.as_ptr() as usize, n, k, buf_tag(b));
    {
        let guard = pack_cache().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = guard.packs.get(&key) {
            return Arc::clone(p);
        }
    }
    let packed = Arc::new(pack_b_amx(b, n, k));
    let mut guard = pack_cache().lock().unwrap_or_else(|e| e.into_inner());
    if guard.packs.len() >= 96 {
        guard.packs.clear();
    }
    guard.packs.insert(key, Arc::clone(&packed));
    packed
}

fn cached_sum_cols(b: &[u8], k: usize, n: usize, signed: bool) -> Arc<Vec<i32>> {
    let key = (b.as_ptr() as usize, n, k, signed, buf_tag(b));
    {
        let guard = pack_cache().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = guard.sums.get(&key) {
            return Arc::clone(s);
        }
    }
    let sums = Arc::new(sum_cols_orig(b, k, n, signed));
    let mut guard = pack_cache().lock().unwrap_or_else(|e| e.into_inner());
    if guard.sums.len() >= 96 {
        guard.sums.clear();
    }
    guard.sums.insert(key, Arc::clone(&sums));
    sums
}

/// B[k, n] → n-tiles of AMX B (16 rows × 64 bytes, K-groups of 4).
fn pack_b_amx(b: &[u8], n: usize, k: usize) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("sse2") {
        // SAFETY: sse2 is baseline on x86_64; loads are 16B aligned in N.
        return unsafe { pack_b_amx_sse2(b, n, k) };
    }
    pack_b_amx_scalar(b, n, k)
}

fn pack_b_amx_scalar(b: &[u8], n: usize, k: usize) -> Vec<u8> {
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

/// Interleave 4 K-rows × 16 N into the AMX B tile (16 groups of 4).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn pack_b_amx_sse2(b: &[u8], n: usize, k: usize) -> Vec<u8> {
    use core::arch::x86_64::*;
    let n_tiles = n / TN;
    let k_tiles = k / TK;
    let mut out = vec![0u8; n_tiles * k_tiles * TM * TK];
    unsafe {
        for nt in 0..n_tiles {
            let n0 = nt * TN;
            for kt in 0..k_tiles {
                let k0 = kt * TK;
                let tile = out.as_mut_ptr().add((nt * k_tiles + kt) * TM * TK);
                for kg in 0..TM {
                    let base = k0 + kg * 4;
                    let r0 = _mm_loadu_si128(b.as_ptr().add(base * n + n0).cast());
                    let r1 = _mm_loadu_si128(b.as_ptr().add((base + 1) * n + n0).cast());
                    let r2 = _mm_loadu_si128(b.as_ptr().add((base + 2) * n + n0).cast());
                    let r3 = _mm_loadu_si128(b.as_ptr().add((base + 3) * n + n0).cast());
                    let a = _mm_unpacklo_epi8(r0, r1);
                    let bb = _mm_unpackhi_epi8(r0, r1);
                    let c = _mm_unpacklo_epi8(r2, r3);
                    let d = _mm_unpackhi_epi8(r2, r3);
                    let e = _mm_unpacklo_epi16(a, c);
                    let f = _mm_unpackhi_epi16(a, c);
                    let g = _mm_unpacklo_epi16(bb, d);
                    let h = _mm_unpackhi_epi16(bb, d);
                    let dst = tile.add(kg * TK);
                    _mm_storeu_si128(dst.cast(), e);
                    _mm_storeu_si128(dst.add(16).cast(), f);
                    _mm_storeu_si128(dst.add(32).cast(), g);
                    _mm_storeu_si128(dst.add(48).cast(), h);
                }
            }
        }
    }
    out
}

fn gemm_tiles(a: &[u8], m0: usize, mb: usize, n: usize, k: usize, packed_b: &[u8], c: &mut [i32]) {
    debug_assert!(mb.is_multiple_of(TM), "AMX path requires M multiple of 16");
    let _amx = amx_lock().lock().unwrap_or_else(|e| e.into_inner());

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
                // No `nomem`: LLVM must not hoist this across `tilestored`.
                core::arch::asm!("tilezero tmm0", options(nostack));
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
        core::arch::asm!("tilerelease", options(nostack));
    }
}
