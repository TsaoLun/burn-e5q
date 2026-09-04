//! D=32 AVX-512 flash for e5-like attention.
//!
//! The generic flash path calls `gemm::gemm` on tiles with K=32. Isolated e5
//! 512 timing showed ~12 GOPS there, plus a scalar `exp` over 12×12×512×512
//! scores (~208 ms, half of `forward_raw`). This kernel keeps the same tiled
//! online-softmax algorithm (TILE=64, no `[S,S]` materialization) but:
//!
//! 1. QK: transpose each K-tile to `[32, 64]` and FMA along the KV axis
//! 2. Softmax: AVX-512 max / Cephes `exp` / sum on full 64-wide tiles
//! 3. PV: two-register D=32 accumulate (same idea as `attention_int8`)
//!
//! Gated to long f32 MHA/GQA with `head_dim == val_dim == 32` and no softcap.
//! Short-seq flash unit tests stay on the bit-close gemm path.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::DType;
use burn_backend::ops::AttentionModuleOptions;
use burn_std::Bytes;

use crate::ops::attention::broadcast_attn_mask_bias;
use crate::{FlexTensor, Layout};

const D: usize = 32;
const TILE: usize = 64;
const MIN_SEQ: usize = 64;

pub(crate) fn should_use(
    query: &FlexTensor,
    key: &FlexTensor,
    value: &FlexTensor,
    options: &AttentionModuleOptions,
) -> bool {
    if options.softcap.is_some() {
        return false;
    }
    if query.dtype() != DType::F32 || key.dtype() != DType::F32 || value.dtype() != DType::F32 {
        return false;
    }
    if query.layout().shape().num_dims() != 4
        || key.layout().shape().num_dims() != 4
        || value.layout().shape().num_dims() != 4
    {
        return false;
    }
    let q = query.layout().shape();
    let k = key.layout().shape();
    let v = value.layout().shape();
    if q[3] != D || k[3] != D || v[3] != D {
        return false;
    }
    if q[2] < MIN_SEQ || k[2] < MIN_SEQ {
        return false;
    }
    avx512f_available()
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

pub(crate) fn flash_d32(
    query: FlexTensor,
    key: FlexTensor,
    value: FlexTensor,
    mask: Option<FlexTensor>,
    attn_bias: Option<FlexTensor>,
    options: AttentionModuleOptions,
) -> FlexTensor {
    debug_assert!(should_use(&query, &key, &value, &options));

    let query = query.to_contiguous();
    let key = key.to_contiguous();
    let value = value.to_contiguous();

    let q_shape = query.layout().shape();
    let k_shape = key.layout().shape();
    let v_shape = value.layout().shape();

    let batch = q_shape[0];
    let heads = q_shape[1];
    let seq_q = q_shape[2];
    let head_dim = q_shape[3];
    let seq_kv = k_shape[2];
    let val_dim = v_shape[3];
    let kv_heads = k_shape[1];

    debug_assert_eq!(head_dim, D);
    debug_assert_eq!(val_dim, D);
    debug_assert!(kv_heads > 0 && heads.is_multiple_of(kv_heads));
    let q_per_kv = heads / kv_heads;

    let target = [batch, heads, seq_q, seq_kv];
    let mask_bcast = mask.map(|m| broadcast_attn_mask_bias(m, target, "mask"));
    let bias_bcast = attn_bias.map(|b| broadcast_attn_mask_bias(b, target, "bias"));

    let scale = options
        .scale
        .unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt()) as f32;
    let causal_offset = if options.is_causal {
        Some(seq_kv as isize - seq_q as isize)
    } else {
        None
    };

    let q_data: &[f32] = query.storage();
    let k_data: &[f32] = key.storage();
    let v_data: &[f32] = value.storage();
    let mask_data: Option<&[u8]> = mask_bcast.as_ref().map(|b| b.tensor.bytes());
    let bias_data: Option<&[f32]> = bias_bcast.as_ref().map(|b| b.tensor.storage());
    let (mask_batch_step, mask_head_step, mask_q_step, mask_tile_len) = mask_bcast
        .as_ref()
        .map(|b| (b.batch_step, b.head_step, b.q_step, b.tile_len))
        .unwrap_or((0, 0, seq_kv, 0));
    let (bias_batch_step, bias_head_step, bias_q_step, bias_tile_len) = bias_bcast
        .as_ref()
        .map(|b| (b.batch_step, b.head_step, b.q_step, b.tile_len))
        .unwrap_or((0, 0, seq_kv, 0));

    let q_head_stride = seq_q * D;
    let q_batch_stride = heads * q_head_stride;
    let k_head_stride = seq_kv * D;
    let k_batch_stride = kv_heads * k_head_stride;
    let v_head_stride = seq_kv * D;
    let v_batch_stride = kv_heads * v_head_stride;
    let o_head_stride = seq_q * D;

    let mut output = vec![0.0f32; batch * heads * seq_q * D];

    let params = HeadParams {
        scale,
        causal_offset,
        seq_q,
        seq_kv,
        q_per_kv,
        q_batch_stride,
        q_head_stride,
        k_batch_stride,
        k_head_stride,
        v_batch_stride,
        v_head_stride,
        mask_batch_step,
        mask_head_step,
        mask_q_step,
        mask_tile_len,
        bias_batch_step,
        bias_head_step,
        bias_q_step,
        bias_tile_len,
    };

    #[cfg(feature = "rayon")]
    if batch * heads > 1 {
        run_heads_rayon(
            batch, heads, o_head_stride, q_data, k_data, v_data, &mut output, mask_data,
            bias_data, &params,
        );
    } else {
        one_head(
            0, 0, q_data, k_data, v_data, &mut output, mask_data, bias_data, &params,
        );
    }

    #[cfg(not(feature = "rayon"))]
    {
        for b in 0..batch {
            for h in 0..heads {
                let o_off = (b * heads + h) * o_head_stride;
                one_head(
                    b,
                    h,
                    q_data,
                    k_data,
                    v_data,
                    &mut output[o_off..o_off + o_head_stride],
                    mask_data,
                    bias_data,
                    &params,
                );
            }
        }
    }

    let shape = burn_std::Shape::from(vec![batch, heads, seq_q, D]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        DType::F32,
    )
}

struct HeadParams {
    scale: f32,
    causal_offset: Option<isize>,
    seq_q: usize,
    seq_kv: usize,
    q_per_kv: usize,
    q_batch_stride: usize,
    q_head_stride: usize,
    k_batch_stride: usize,
    k_head_stride: usize,
    v_batch_stride: usize,
    v_head_stride: usize,
    mask_batch_step: usize,
    mask_head_step: usize,
    mask_q_step: usize,
    mask_tile_len: usize,
    bias_batch_step: usize,
    bias_head_step: usize,
    bias_q_step: usize,
    bias_tile_len: usize,
}

/// Flattened `(batch, head)` work, but only ~one task per CPU.
///
/// A packed e5 forward is `[8, 12, 512, 32]` → 96 heads. One rayon task
/// per head on a 4-core box thrashes L2 and blew the packed batch from
/// ~3.8 s to ~19 s. Group heads so each worker reuses scratch in L1/L2.
#[cfg(feature = "rayon")]
fn run_heads_rayon(
    batch: usize,
    heads: usize,
    o_head_stride: usize,
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    output: &mut [f32],
    mask_data: Option<&[u8]>,
    bias_data: Option<&[f32]>,
    params: &HeadParams,
) {
    use rayon::prelude::*;
    let n = batch * heads;
    let threads = rayon::current_num_threads().max(1);
    let chunk_heads = n.div_ceil(threads).max(1);
    output
        .par_chunks_mut(chunk_heads * o_head_stride)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * chunk_heads;
            for (local, out_head) in out_chunk.chunks_mut(o_head_stride).enumerate() {
                let idx = start + local;
                let b = idx / heads;
                let h = idx % heads;
                one_head(
                    b, h, q_data, k_data, v_data, out_head, mask_data, bias_data, params,
                );
            }
        });
}

fn one_head(
    b: usize,
    h: usize,
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    out_head: &mut [f32],
    mask_data: Option<&[u8]>,
    bias_data: Option<&[f32]>,
    p: &HeadParams,
) {
    let kv_h = h / p.q_per_kv;
    let q_off = b * p.q_batch_stride + h * p.q_head_stride;
    let k_off = b * p.k_batch_stride + kv_h * p.k_head_stride;
    let v_off = b * p.v_batch_stride + kv_h * p.v_head_stride;
    let mask_off = b * p.mask_batch_step + h * p.mask_head_step;
    let bias_off = b * p.bias_batch_step + h * p.bias_head_step;

    flash_head_d32(
        &q_data[q_off..q_off + p.q_head_stride],
        &k_data[k_off..k_off + p.k_head_stride],
        &v_data[v_off..v_off + p.v_head_stride],
        out_head,
        mask_data.map(|m| &m[mask_off..mask_off + p.mask_tile_len]),
        bias_data.map(|bias| &bias[bias_off..bias_off + p.bias_tile_len]),
        p,
    );
}

fn flash_head_d32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    mask: Option<&[u8]>,
    bias: Option<&[f32]>,
    p: &HeadParams,
) {
    let seq_q = p.seq_q;
    let seq_kv = p.seq_kv;
    let scale = p.scale;
    let causal_offset = p.causal_offset;

    let mut row_max = vec![f32::NEG_INFINITY; seq_q];
    let mut row_sum = vec![0.0f32; seq_q];
    let mut scores = vec![0.0f32; seq_q * TILE];
    output.fill(0.0);

    let num_tiles = seq_kv.div_ceil(TILE);
    for tile_idx in 0..num_tiles {
        let kv_start = tile_idx * TILE;
        let tile_kv = (seq_kv - kv_start).min(TILE);

        qk_tile(q, k, kv_start, tile_kv, seq_q, scale, &mut scores);

        for qi in 0..seq_q {
            let row = &mut scores[qi * TILE..qi * TILE + tile_kv];
            apply_mask_bias_causal(
                row,
                qi,
                kv_start,
                mask,
                bias,
                causal_offset,
                p.mask_q_step,
                p.bias_q_step,
            );

            let tile_max = max_slice(row);
            if tile_max == f32::NEG_INFINITY {
                row.fill(0.0);
                continue;
            }

            let new_max = if row_max[qi] > tile_max {
                row_max[qi]
            } else {
                tile_max
            };
            let tile_sum = exp_sub_inplace(row, new_max);
            let correction = if row_max[qi] == f32::NEG_INFINITY {
                0.0
            } else {
                (row_max[qi] - new_max).exp()
            };

            let out_row = &mut output[qi * D..qi * D + D];
            scale_row32(out_row, correction);
            row_sum[qi] = row_sum[qi] * correction + tile_sum;
            row_max[qi] = new_max;
        }

        pv_tile(v, kv_start, tile_kv, seq_q, &scores, output);
    }

    for qi in 0..seq_q {
        let sum = row_sum[qi];
        if sum > 0.0 {
            scale_row32(&mut output[qi * D..qi * D + D], 1.0 / sum);
        }
    }
}

fn apply_mask_bias_causal(
    row: &mut [f32],
    qi: usize,
    kv_start: usize,
    mask: Option<&[u8]>,
    bias: Option<&[f32]>,
    causal_offset: Option<isize>,
    mask_q_step: usize,
    bias_q_step: usize,
) {
    if mask.is_none() && bias.is_none() && causal_offset.is_none() {
        return;
    }
    for (ki, val) in row.iter_mut().enumerate() {
        let kv_idx = kv_start + ki;
        if let Some(m) = mask
            && m[qi * mask_q_step + kv_idx] != 0
        {
            *val = f32::NEG_INFINITY;
            continue;
        }
        if let Some(offset) = causal_offset
            && (kv_idx as isize) > (qi as isize) + offset
        {
            *val = f32::NEG_INFINITY;
            continue;
        }
        if let Some(b) = bias {
            *val += b[qi * bias_q_step + kv_idx];
        }
    }
}

fn max_slice(row: &[f32]) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for &v in row {
        if v > m {
            m = v;
        }
    }
    m
}

fn exp_sub_inplace(row: &mut [f32], max: f32) -> f32 {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if row.len() == TILE && avx512f_available() {
        // SAFETY: avx512f checked; `row` is TILE f32s.
        return unsafe { exp_sub_inplace_avx512_64(row, max) };
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += e;
    }
    sum
}

fn scale_row32(row: &mut [f32], scale: f32) {
    debug_assert_eq!(row.len(), D);
    if scale == 1.0 {
        return;
    }
    if scale == 0.0 {
        row.fill(0.0);
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if avx512f_available() {
        // SAFETY: avx512f checked; `row` is 32 f32s.
        unsafe {
            scale_row32_avx512(row, scale);
        }
        return;
    }
    for x in row.iter_mut() {
        *x *= scale;
    }
}

fn qk_tile(
    q: &[f32],
    k: &[f32],
    kv_start: usize,
    tile_kv: usize,
    seq_q: usize,
    scale: f32,
    scores: &mut [f32],
) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if avx512f_available() {
        // SAFETY: avx512f checked.
        unsafe {
            qk_tile_avx512(q, k, kv_start, tile_kv, seq_q, scale, scores);
        }
        return;
    }
    qk_tile_scalar(q, k, kv_start, tile_kv, seq_q, scale, scores);
}

fn qk_tile_scalar(
    q: &[f32],
    k: &[f32],
    kv_start: usize,
    tile_kv: usize,
    seq_q: usize,
    scale: f32,
    scores: &mut [f32],
) {
    for qi in 0..seq_q {
        let qrow = &q[qi * D..qi * D + D];
        let dest = &mut scores[qi * TILE..qi * TILE + tile_kv];
        for ki in 0..tile_kv {
            let krow = &k[(kv_start + ki) * D..(kv_start + ki) * D + D];
            let mut acc = 0.0f32;
            for d in 0..D {
                acc += qrow[d] * krow[d];
            }
            dest[ki] = acc * scale;
        }
    }
}

fn pv_tile(
    v: &[f32],
    kv_start: usize,
    tile_kv: usize,
    seq_q: usize,
    scores: &[f32],
    output: &mut [f32],
) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if avx512f_available() {
        // SAFETY: avx512f checked.
        unsafe {
            pv_tile_avx512(v, kv_start, tile_kv, seq_q, scores, output);
        }
        return;
    }
    for qi in 0..seq_q {
        let prow = &scores[qi * TILE..qi * TILE + tile_kv];
        let orow = &mut output[qi * D..qi * D + D];
        for ki in 0..tile_kv {
            let p = prow[ki];
            if p == 0.0 {
                continue;
            }
            let vrow = &v[(kv_start + ki) * D..(kv_start + ki) * D + D];
            for d in 0..D {
                orow[d] += p * vrow[d];
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn qk_tile_avx512(
    q: &[f32],
    k: &[f32],
    kv_start: usize,
    tile_kv: usize,
    seq_q: usize,
    scale: f32,
    scores: &mut [f32],
) {
    use core::arch::x86_64::*;

    unsafe {
        // K-tile as [D, TILE] so the inner FMA walks consecutive KV lanes.
        let mut k_t = [0.0f32; D * TILE];
        for ki in 0..tile_kv {
            let krow = k.as_ptr().add((kv_start + ki) * D);
            for d in 0..D {
                k_t[d * TILE + ki] = *krow.add(d);
            }
        }

        let vscale = _mm512_set1_ps(scale);
        for qi in 0..seq_q {
            let qrow = q.as_ptr().add(qi * D);
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();
            for d in 0..D {
                let qd = _mm512_set1_ps(*qrow.add(d));
                let kt = k_t.as_ptr().add(d * TILE);
                acc0 = _mm512_fmadd_ps(qd, _mm512_loadu_ps(kt), acc0);
                acc1 = _mm512_fmadd_ps(qd, _mm512_loadu_ps(kt.add(16)), acc1);
                acc2 = _mm512_fmadd_ps(qd, _mm512_loadu_ps(kt.add(32)), acc2);
                acc3 = _mm512_fmadd_ps(qd, _mm512_loadu_ps(kt.add(48)), acc3);
            }
            acc0 = _mm512_mul_ps(acc0, vscale);
            acc1 = _mm512_mul_ps(acc1, vscale);
            acc2 = _mm512_mul_ps(acc2, vscale);
            acc3 = _mm512_mul_ps(acc3, vscale);

            let dest = scores.as_mut_ptr().add(qi * TILE);
            if tile_kv == TILE {
                _mm512_storeu_ps(dest, acc0);
                _mm512_storeu_ps(dest.add(16), acc1);
                _mm512_storeu_ps(dest.add(32), acc2);
                _mm512_storeu_ps(dest.add(48), acc3);
            } else {
                let mut tmp = [0.0f32; TILE];
                _mm512_storeu_ps(tmp.as_mut_ptr(), acc0);
                _mm512_storeu_ps(tmp.as_mut_ptr().add(16), acc1);
                _mm512_storeu_ps(tmp.as_mut_ptr().add(32), acc2);
                _mm512_storeu_ps(tmp.as_mut_ptr().add(48), acc3);
                core::ptr::copy_nonoverlapping(tmp.as_ptr(), dest, tile_kv);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn pv_tile_avx512(
    v: &[f32],
    kv_start: usize,
    tile_kv: usize,
    seq_q: usize,
    scores: &[f32],
    output: &mut [f32],
) {
    use core::arch::x86_64::*;
    unsafe {
        for qi in 0..seq_q {
            let prow = scores.as_ptr().add(qi * TILE);
            let optr = output.as_mut_ptr().add(qi * D);
            let mut acc0 = _mm512_loadu_ps(optr);
            let mut acc1 = _mm512_loadu_ps(optr.add(16));
            for ki in 0..tile_kv {
                let p = *prow.add(ki);
                if p == 0.0 {
                    continue;
                }
                let pv = _mm512_set1_ps(p);
                let vptr = v.as_ptr().add((kv_start + ki) * D);
                acc0 = _mm512_fmadd_ps(pv, _mm512_loadu_ps(vptr), acc0);
                acc1 = _mm512_fmadd_ps(pv, _mm512_loadu_ps(vptr.add(16)), acc1);
            }
            _mm512_storeu_ps(optr, acc0);
            _mm512_storeu_ps(optr.add(16), acc1);
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn scale_row32_avx512(row: &mut [f32], scale: f32) {
    use core::arch::x86_64::*;
    unsafe {
        let s = _mm512_set1_ps(scale);
        let p = row.as_mut_ptr();
        _mm512_storeu_ps(p, _mm512_mul_ps(_mm512_loadu_ps(p), s));
        _mm512_storeu_ps(p.add(16), _mm512_mul_ps(_mm512_loadu_ps(p.add(16)), s));
    }
}

/// Cephes `expf` on 16 lanes (avx_mathfun constants) + `_mm512_scalef_ps`.
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

        // r = x - n * ln2, split ln2 for extra bits.
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693_359_375), x);
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.121_944_40e-4), r);

        // cephes exp polynomial in r, then * r^2 + r + 1.
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

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn exp_sub_inplace_avx512_64(row: &mut [f32], max: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let p = row.as_mut_ptr();
        let vmax = _mm512_set1_ps(max);
        let mut sum = _mm512_setzero_ps();
        for off in [0, 16, 32, 48] {
            let x = _mm512_sub_ps(_mm512_loadu_ps(p.add(off)), vmax);
            let e = exp_ps_avx512(x);
            _mm512_storeu_ps(p.add(off), e);
            sum = _mm512_add_ps(sum, e);
        }
        _mm512_reduce_add_ps(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::attention::{attention_flash_gemm, attention_naive};
    use burn_std::Shape;

    fn flex_f32(data: Vec<f32>, shape: &[usize]) -> FlexTensor {
        FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            DType::F32,
        )
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for (&x, &y) in a.iter().zip(b) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt()).max(1e-12)
    }

    fn make_e5_like(batch: usize, heads: usize, seq: usize) -> (FlexTensor, FlexTensor, FlexTensor) {
        let n = batch * heads * seq * D;
        let q = flex_f32(
            (0..n).map(|i| ((i % 97) as f32) * 0.02 - 0.7).collect(),
            &[batch, heads, seq, D],
        );
        let k = flex_f32(
            (0..n).map(|i| ((i % 89) as f32) * 0.017 - 0.4).collect(),
            &[batch, heads, seq, D],
        );
        let v = flex_f32(
            (0..n).map(|i| ((i % 83) as f32) * 0.013 - 0.2).collect(),
            &[batch, heads, seq, D],
        );
        (q, k, v)
    }

    #[test]
    fn d32_stays_off_for_short_seq() {
        let q = flex_f32(vec![1.0; 1 * 1 * 8 * D], &[1, 1, 8, D]);
        assert!(!should_use(&q, &q, &q, &Default::default()));
    }

    #[test]
    fn d32_flash_close_to_naive_and_gemm_flash() {
        if !avx512f_available() {
            return;
        }
        let (q, k, v) = make_e5_like(1, 2, 128);
        assert!(should_use(&q, &k, &v, &Default::default()));

        let d32 = flash_d32(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            None,
            Default::default(),
        );
        let naive = attention_naive(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            None,
            Default::default(),
        );
        let gemm = attention_flash_gemm(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            None,
            Default::default(),
        );
        let da: &[f32] = d32.storage();
        let na: &[f32] = naive.storage();
        let ga: &[f32] = gemm.storage();
        let cos_n = cosine(da, na);
        let cos_g = cosine(da, ga);
        assert!(
            cos_n > 0.999,
            "d32 vs naive cosine {cos_n} (expected > 0.999)"
        );
        assert!(
            cos_g > 0.999,
            "d32 vs gemm-flash cosine {cos_g} (expected > 0.999)"
        );
    }

    #[test]
    fn d32_flash_partial_tile_and_bias() {
        if !avx512f_available() {
            return;
        }
        let seq = 100;
        let (q, k, v) = make_e5_like(1, 1, seq);
        let bias = flex_f32(
            (0..seq).map(|i| if i < 10 { -1.0e4 } else { 0.0 }).collect(),
            &[1, 1, 1, seq],
        );
        let d32 = flash_d32(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            Some(bias.clone()),
            Default::default(),
        );
        let naive = attention_naive(q, k, v, None, Some(bias), Default::default());
        let cos = cosine(d32.storage(), naive.storage());
        assert!(cos > 0.999, "partial-tile+bias cosine {cos}");
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn d32_exp_tracks_libm() {
        if !avx512f_available() {
            return;
        }
        let mut xs: [f32; 64] = core::array::from_fn(|i| -0.2 * (i as f32));
        let expected: Vec<f32> = xs.iter().map(|&x| (x - 0.0).exp()).collect();
        // SAFETY: test is gated on avx512f.
        let sum = unsafe { exp_sub_inplace_avx512_64(&mut xs, 0.0) };
        let mut max_rel = 0.0f32;
        for (a, b) in xs.iter().zip(&expected) {
            let rel = (a - b).abs() / b.abs().max(1e-12);
            if rel > max_rel {
                max_rel = rel;
            }
        }
        assert!(
            max_rel < 2e-5,
            "cephes exp relative error {max_rel} vs libm"
        );
        let exp_sum: f32 = expected.iter().sum();
        assert!((sum - exp_sum).abs() / exp_sum.max(1e-12) < 2e-5);
    }

    #[test]
    fn d32_flash_faster_than_gemm_e5_512() {
        if !avx512f_available() {
            return;
        }
        let (q, k, v) = make_e5_like(1, 12, 512);
        assert!(should_use(&q, &k, &v, &Default::default()));

        let time_ms = |f: &mut dyn FnMut()| {
            let start = std::time::Instant::now();
            f();
            start.elapsed().as_secs_f64() * 1e3
        };

        // Warmup
        let _ = flash_d32(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            None,
            Default::default(),
        );
        let _ = attention_flash_gemm(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            None,
            Default::default(),
        );

        let mut d32_ms = f64::INFINITY;
        let mut gemm_ms = f64::INFINITY;
        for _ in 0..3 {
            d32_ms = d32_ms.min(time_ms(&mut || {
                let _ = flash_d32(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    None,
                    None,
                    Default::default(),
                );
            }));
            gemm_ms = gemm_ms.min(time_ms(&mut || {
                let _ = attention_flash_gemm(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    None,
                    None,
                    Default::default(),
                );
            }));
        }
        // 12 layers like e5.
        let bias_row = flex_f32(vec![0.0f32; 512], &[1, 1, 1, 512]);
        let mut d32_bias_ms = f64::INFINITY;
        let mut gemm_bias_ms = f64::INFINITY;
        for _ in 0..3 {
            d32_bias_ms = d32_bias_ms.min(time_ms(&mut || {
                let _ = flash_d32(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    None,
                    Some(bias_row.clone()),
                    Default::default(),
                );
            }));
            gemm_bias_ms = gemm_bias_ms.min(time_ms(&mut || {
                let _ = attention_flash_gemm(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    None,
                    Some(bias_row.clone()),
                    Default::default(),
                );
            }));
        }
        println!(
            "e5-like 12h×512: d32 {d32_ms:.1} ms / layer, gemm-flash {gemm_ms:.1} ms / layer; ×12 ≈ {:.0} vs {:.0}",
            d32_ms * 12.0,
            gemm_ms * 12.0
        );
        println!(
            "same + bias[1,1,1,S]: d32 {d32_bias_ms:.1} ms / layer, gemm-flash {gemm_bias_ms:.1} ms / layer; ×12 ≈ {:.0} vs {:.0}",
            d32_bias_ms * 12.0,
            gemm_bias_ms * 12.0
        );
        assert!(
            d32_ms < gemm_ms * 0.85,
            "d32 {d32_ms:.1} ms should beat gemm-flash {gemm_ms:.1} ms by >15%"
        );

        // Packed compare_ort is `[8, 12, 512, 32]`. 96 one-head rayon tasks
        // were ~5× slower than 8× the B=1 kernel; grain must stay near linear.
        let (q8, k8, v8) = make_e5_like(8, 12, 512);
        let mut d32_b8 = f64::INFINITY;
        for _ in 0..2 {
            d32_b8 = d32_b8.min(time_ms(&mut || {
                let _ = flash_d32(
                    q8.clone(),
                    k8.clone(),
                    v8.clone(),
                    None,
                    None,
                    Default::default(),
                );
            }));
        }
        println!(
            "e5-like 8×12h×512: d32 {d32_b8:.1} ms ({:.1}× the 1×12h kernel)",
            d32_b8 / d32_ms.max(1e-9)
        );
        assert!(
            d32_b8 < d32_ms * 16.0,
            "B=8 d32 {d32_b8:.1} ms should stay under 16× B=1 {d32_ms:.1} ms"
        );
    }
}
