//! D=32 AVX-512 flash for e5-like attention.
//!
//! The generic flash path calls `gemm::gemm` on tiles with K=32. Isolated e5
//! 512 timing showed ~12 GOPS there, plus a scalar `exp` over 12×12×512×512
//! scores (~208 ms, half of `forward_raw`). This kernel keeps the same tiled
//! online-softmax algorithm (TILE=64, no `[S,S]` materialization) but:
//!
//! 1. QK: transpose each K-tile to `[32, 64]` once, then FMA along the KV axis
//! 2. Softmax: AVX-512 max / Cephes `exp` / sum on full 64-wide tiles
//! 3. PV: two-register D=32 accumulate (same idea as `attention_int8`)
//! 4. Q-block (BR=16): scores stay in L1; K transpose is hoisted per KV tile
//!
//! TILE is still 64 (KV / Bc). BR is the query block (Br), not a TILE change.
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
/// Query block (FlashAttention Br). TILE is still the KV tile (Bc).
const BR: usize = 16;
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
            batch,
            heads,
            o_head_stride,
            q_data,
            k_data,
            v_data,
            &mut output,
            mask_data,
            bias_data,
            &params,
        );
    } else {
        let mut scratch = FlashScratch::with_seq(seq_q);
        one_head(
            0,
            0,
            q_data,
            k_data,
            v_data,
            &mut output,
            mask_data,
            bias_data,
            &params,
            &mut scratch,
        );
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut scratch = FlashScratch::with_seq(seq_q);
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
                    &mut scratch,
                );
                scratch.reset();
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

struct FlashScratch {
    row_max: Vec<f32>,
    row_sum: Vec<f32>,
    scores: Vec<f32>,
}

impl FlashScratch {
    fn with_seq(seq_q: usize) -> Self {
        Self {
            row_max: vec![f32::NEG_INFINITY; seq_q],
            row_sum: vec![0.0f32; seq_q],
            scores: vec![0.0f32; BR * TILE],
        }
    }

    fn reset(&mut self) {
        self.row_max.fill(f32::NEG_INFINITY);
        self.row_sum.fill(0.0);
    }
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
            let mut scratch = FlashScratch::with_seq(params.seq_q);
            for (local, out_head) in out_chunk.chunks_mut(o_head_stride).enumerate() {
                let idx = start + local;
                let b = idx / heads;
                let h = idx % heads;
                one_head(
                    b,
                    h,
                    q_data,
                    k_data,
                    v_data,
                    out_head,
                    mask_data,
                    bias_data,
                    params,
                    &mut scratch,
                );
                scratch.reset();
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
    scratch: &mut FlashScratch,
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
        scratch,
    );
}

/// Per-thread: flush denormals. Padded e5 rows (6 shorts in a `[7,512]` pack)
/// otherwise spend tens of seconds in denormal AVX-512 QK/PV.
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

fn flash_head_d32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    mask: Option<&[u8]>,
    bias: Option<&[f32]>,
    p: &HeadParams,
    scratch: &mut FlashScratch,
) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    enable_ftz_daz();

    let seq_q = p.seq_q;
    let seq_kv = p.seq_kv;
    let scale = p.scale;
    let causal_offset = p.causal_offset;

    let row_max = &mut scratch.row_max;
    let row_sum = &mut scratch.row_sum;
    let scores = &mut scratch.scores;
    output.fill(0.0);

    // e5 bias is `[B,1,1,S]` (`q_step == 0`): add it once in QK instead of
    // a scalar sweep over every query row.
    let fuse_bias = bias.is_some() && p.bias_q_step == 0;
    let qk_bias = if fuse_bias { bias } else { None };
    let row_bias = if fuse_bias { None } else { bias };

    let mut k_t = [0.0f32; D * TILE];
    let num_tiles = seq_kv.div_ceil(TILE);
    for tile_idx in 0..num_tiles {
        let kv_start = tile_idx * TILE;
        let tile_kv = (seq_kv - kv_start).min(TILE);
        transpose_k_tile(k, kv_start, tile_kv, &mut k_t);

        // Br=16: scores are BR×TILE (4 KiB) and stay in L1. TILE (Bc) is
        // still 64. Walking all 512 queries against one K-tile used to
        // write a 128 KiB score panel before softmax/PV reread it.
        let mut q0 = 0;
        while q0 < seq_q {
            let bq = (seq_q - q0).min(BR);
            let score_block = &mut scores[..bq * TILE];
            qk_block(
                &q[q0 * D..(q0 + bq) * D],
                &k_t,
                kv_start,
                tile_kv,
                bq,
                scale,
                score_block,
                qk_bias,
            );

            let e5_softmax =
                tile_kv == TILE && mask.is_none() && row_bias.is_none() && causal_offset.is_none();
            #[cfg(all(target_arch = "x86_64", feature = "std"))]
            let e5_softmax = e5_softmax && avx512f_available();
            #[cfg(all(target_arch = "x86_64", feature = "std"))]
            if e5_softmax {
                let mut qi = 0;
                while qi + 4 <= bq {
                    // SAFETY: avx512f checked; four full TILE score rows.
                    unsafe {
                        online_softmax_4(
                            &mut score_block[qi * TILE..(qi + 4) * TILE],
                            q0 + qi,
                            row_max,
                            row_sum,
                            output,
                        );
                    }
                    qi += 4;
                }
                while qi < bq {
                    online_softmax_one(
                        &mut score_block[qi * TILE..qi * TILE + tile_kv],
                        q0 + qi,
                        row_max,
                        row_sum,
                        output,
                        mask,
                        row_bias,
                        causal_offset,
                        kv_start,
                        p,
                    );
                    qi += 1;
                }
            } else {
                for qi in 0..bq {
                    online_softmax_one(
                        &mut score_block[qi * TILE..qi * TILE + tile_kv],
                        q0 + qi,
                        row_max,
                        row_sum,
                        output,
                        mask,
                        row_bias,
                        causal_offset,
                        kv_start,
                        p,
                    );
                }
            }
            #[cfg(not(all(target_arch = "x86_64", feature = "std")))]
            {
                let _ = e5_softmax;
                for qi in 0..bq {
                    online_softmax_one(
                        &mut score_block[qi * TILE..qi * TILE + tile_kv],
                        q0 + qi,
                        row_max,
                        row_sum,
                        output,
                        mask,
                        row_bias,
                        causal_offset,
                        kv_start,
                        p,
                    );
                }
            }

            pv_block(
                v,
                kv_start,
                tile_kv,
                bq,
                score_block,
                &mut output[q0 * D..(q0 + bq) * D],
            );
            q0 += bq;
        }
    }

    for qi in 0..seq_q {
        let sum = row_sum[qi];
        if sum > 0.0 {
            scale_row32(&mut output[qi * D..qi * D + D], 1.0 / sum);
        }
    }
}

fn online_softmax_one(
    row: &mut [f32],
    gqi: usize,
    row_max: &mut [f32],
    row_sum: &mut [f32],
    output: &mut [f32],
    mask: Option<&[u8]>,
    row_bias: Option<&[f32]>,
    causal_offset: Option<isize>,
    kv_start: usize,
    p: &HeadParams,
) {
    apply_mask_bias_causal(
        row,
        gqi,
        kv_start,
        mask,
        row_bias,
        causal_offset,
        p.mask_q_step,
        p.bias_q_step,
    );

    let tile_max = max_slice(row);
    if tile_max == f32::NEG_INFINITY {
        row.fill(0.0);
        return;
    }

    let new_max = if row_max[gqi] > tile_max {
        row_max[gqi]
    } else {
        tile_max
    };
    let tile_sum = exp_sub_inplace(row, new_max);
    let correction = if row_max[gqi] == f32::NEG_INFINITY {
        0.0
    } else {
        (row_max[gqi] - new_max).exp()
    };

    let out_row = &mut output[gqi * D..gqi * D + D];
    scale_row32(out_row, correction);
    row_sum[gqi] = row_sum[gqi] * correction + tile_sum;
    row_max[gqi] = new_max;
}

/// Four full TILE rows: interleave max / Cephes exp / output rescale so
/// the long `exp` latency of one row overlaps the others. Same arithmetic
/// as [`online_softmax_one`].
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn online_softmax_4(
    scores: &mut [f32],
    q0: usize,
    row_max: &mut [f32],
    row_sum: &mut [f32],
    output: &mut [f32],
) {
    debug_assert_eq!(scores.len(), 4 * TILE);
    unsafe {
        let m0 = max_slice_avx512_64(&scores[0..TILE]);
        let m1 = max_slice_avx512_64(&scores[TILE..2 * TILE]);
        let m2 = max_slice_avx512_64(&scores[2 * TILE..3 * TILE]);
        let m3 = max_slice_avx512_64(&scores[3 * TILE..4 * TILE]);
        // Match `online_softmax_one`: a fully-masked row must not
        // `exp(x - (-inf))`. e5 has no mask; keep the interleaved
        // exp/scale on the finite-max hot path.
        if m0 == f32::NEG_INFINITY
            || m1 == f32::NEG_INFINITY
            || m2 == f32::NEG_INFINITY
            || m3 == f32::NEG_INFINITY
        {
            softmax4_one_row(&mut scores[0..TILE], q0, m0, row_max, row_sum, output);
            softmax4_one_row(
                &mut scores[TILE..2 * TILE],
                q0 + 1,
                m1,
                row_max,
                row_sum,
                output,
            );
            softmax4_one_row(
                &mut scores[2 * TILE..3 * TILE],
                q0 + 2,
                m2,
                row_max,
                row_sum,
                output,
            );
            softmax4_one_row(
                &mut scores[3 * TILE..4 * TILE],
                q0 + 3,
                m3,
                row_max,
                row_sum,
                output,
            );
            return;
        }
        let (n0, c0) = online_new_max(row_max[q0], m0);
        let (n1, c1) = online_new_max(row_max[q0 + 1], m1);
        let (n2, c2) = online_new_max(row_max[q0 + 2], m2);
        let (n3, c3) = online_new_max(row_max[q0 + 3], m3);
        let s0 = exp_sub_inplace_avx512_64(&mut scores[0..TILE], n0);
        let s1 = exp_sub_inplace_avx512_64(&mut scores[TILE..2 * TILE], n1);
        let s2 = exp_sub_inplace_avx512_64(&mut scores[2 * TILE..3 * TILE], n2);
        let s3 = exp_sub_inplace_avx512_64(&mut scores[3 * TILE..4 * TILE], n3);
        scale_row32(&mut output[q0 * D..(q0 + 1) * D], c0);
        scale_row32(&mut output[(q0 + 1) * D..(q0 + 2) * D], c1);
        scale_row32(&mut output[(q0 + 2) * D..(q0 + 3) * D], c2);
        scale_row32(&mut output[(q0 + 3) * D..(q0 + 4) * D], c3);
        row_sum[q0] = row_sum[q0] * c0 + s0;
        row_sum[q0 + 1] = row_sum[q0 + 1] * c1 + s1;
        row_sum[q0 + 2] = row_sum[q0 + 2] * c2 + s2;
        row_sum[q0 + 3] = row_sum[q0 + 3] * c3 + s3;
        row_max[q0] = n0;
        row_max[q0 + 1] = n1;
        row_max[q0 + 2] = n2;
        row_max[q0 + 3] = n3;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn softmax4_one_row(
    row: &mut [f32],
    gqi: usize,
    tile_max: f32,
    row_max: &mut [f32],
    row_sum: &mut [f32],
    output: &mut [f32],
) {
    if tile_max == f32::NEG_INFINITY {
        row.fill(0.0);
        return;
    }
    let (new_max, correction) = online_new_max(row_max[gqi], tile_max);
    let tile_sum = unsafe { exp_sub_inplace_avx512_64(row, new_max) };
    scale_row32(&mut output[gqi * D..gqi * D + D], correction);
    row_sum[gqi] = row_sum[gqi] * correction + tile_sum;
    row_max[gqi] = new_max;
}

fn online_new_max(prev: f32, tile_max: f32) -> (f32, f32) {
    if tile_max == f32::NEG_INFINITY {
        return (prev, 1.0);
    }
    let new_max = if prev > tile_max { prev } else { tile_max };
    let correction = if prev == f32::NEG_INFINITY {
        0.0
    } else {
        (prev - new_max).exp()
    };
    (new_max, correction)
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
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if row.len() == TILE && avx512f_available() {
        // SAFETY: avx512f checked; `row` is TILE f32s.
        return unsafe { max_slice_avx512_64(row) };
    }
    let mut m = f32::NEG_INFINITY;
    for &v in row {
        if v > m {
            m = v;
        }
    }
    m
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn max_slice_avx512_64(row: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let p = row.as_ptr();
        let m = _mm512_max_ps(_mm512_loadu_ps(p), _mm512_loadu_ps(p.add(16)));
        let m = _mm512_max_ps(m, _mm512_loadu_ps(p.add(32)));
        let m = _mm512_max_ps(m, _mm512_loadu_ps(p.add(48)));
        _mm512_reduce_max_ps(m)
    }
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

fn transpose_k_tile(k: &[f32], kv_start: usize, tile_kv: usize, k_t: &mut [f32; D * TILE]) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if tile_kv == TILE && avx512f_available() {
        // SAFETY: avx512f implies AVX; 8×8 ymm transpose of a full TILE.
        unsafe { transpose_k_tile_avx(k, kv_start, k_t) };
        return;
    }
    if tile_kv < TILE {
        k_t.fill(0.0);
    }
    for ki in 0..tile_kv {
        let krow = &k[(kv_start + ki) * D..];
        for d in 0..D {
            k_t[d * TILE + ki] = krow[d];
        }
    }
}

/// 8×8 AVX transpose of each `[8, 8]` block in a `[64, 32]` K tile.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx")]
unsafe fn transpose_k_tile_avx(k: &[f32], kv_start: usize, k_t: &mut [f32; D * TILE]) {
    use core::arch::x86_64::*;
    unsafe {
        let kp = k.as_ptr().add(kv_start * D);
        let tp = k_t.as_mut_ptr();
        let mut ki0 = 0;
        while ki0 < TILE {
            let mut d0 = 0;
            while d0 < D {
                let mut r = [_mm256_setzero_ps(); 8];
                for i in 0..8 {
                    r[i] = _mm256_loadu_ps(kp.add((ki0 + i) * D + d0));
                }
                transpose8x8_ps(&mut r);
                for i in 0..8 {
                    _mm256_storeu_ps(tp.add((d0 + i) * TILE + ki0), r[i]);
                }
                d0 += 8;
            }
            ki0 += 8;
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx")]
#[allow(unused_unsafe)]
unsafe fn transpose8x8_ps(r: &mut [core::arch::x86_64::__m256; 8]) {
    use core::arch::x86_64::*;
    unsafe {
        let t0 = _mm256_unpacklo_ps(r[0], r[1]);
        let t1 = _mm256_unpackhi_ps(r[0], r[1]);
        let t2 = _mm256_unpacklo_ps(r[2], r[3]);
        let t3 = _mm256_unpackhi_ps(r[2], r[3]);
        let t4 = _mm256_unpacklo_ps(r[4], r[5]);
        let t5 = _mm256_unpackhi_ps(r[4], r[5]);
        let t6 = _mm256_unpacklo_ps(r[6], r[7]);
        let t7 = _mm256_unpackhi_ps(r[6], r[7]);

        let tt0 = _mm256_shuffle_ps(t0, t2, 0x44);
        let tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
        let tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
        let tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
        let tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
        let tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
        let tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
        let tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

        r[0] = _mm256_permute2f128_ps(tt0, tt4, 0x20);
        r[1] = _mm256_permute2f128_ps(tt1, tt5, 0x20);
        r[2] = _mm256_permute2f128_ps(tt2, tt6, 0x20);
        r[3] = _mm256_permute2f128_ps(tt3, tt7, 0x20);
        r[4] = _mm256_permute2f128_ps(tt0, tt4, 0x31);
        r[5] = _mm256_permute2f128_ps(tt1, tt5, 0x31);
        r[6] = _mm256_permute2f128_ps(tt2, tt6, 0x31);
        r[7] = _mm256_permute2f128_ps(tt3, tt7, 0x31);
    }
}

fn qk_block(
    q: &[f32],
    k_t: &[f32; D * TILE],
    kv_start: usize,
    tile_kv: usize,
    bq: usize,
    scale: f32,
    scores: &mut [f32],
    bias: Option<&[f32]>,
) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if avx512f_available() {
        // SAFETY: avx512f checked.
        unsafe {
            qk_block_avx512(q, k_t, kv_start, tile_kv, bq, scale, scores, bias);
        }
        return;
    }
    qk_block_scalar(q, k_t, kv_start, tile_kv, bq, scale, scores, bias);
}

fn qk_block_scalar(
    q: &[f32],
    k_t: &[f32; D * TILE],
    kv_start: usize,
    tile_kv: usize,
    bq: usize,
    scale: f32,
    scores: &mut [f32],
    bias: Option<&[f32]>,
) {
    for qi in 0..bq {
        let qrow = &q[qi * D..qi * D + D];
        let dest = &mut scores[qi * TILE..qi * TILE + tile_kv];
        for ki in 0..tile_kv {
            let mut acc = 0.0f32;
            for d in 0..D {
                acc += qrow[d] * k_t[d * TILE + ki];
            }
            dest[ki] = acc * scale;
        }
        if let Some(b) = bias {
            for ki in 0..tile_kv {
                dest[ki] += b[kv_start + ki];
            }
        }
    }
}

fn pv_block(
    v: &[f32],
    kv_start: usize,
    tile_kv: usize,
    bq: usize,
    scores: &[f32],
    output: &mut [f32],
) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if avx512f_available() {
        // SAFETY: avx512f checked.
        unsafe {
            pv_block_avx512(v, kv_start, tile_kv, bq, scores, output);
        }
        return;
    }
    for qi in 0..bq {
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
unsafe fn qk_block_avx512(
    q: &[f32],
    k_t: &[f32; D * TILE],
    kv_start: usize,
    tile_kv: usize,
    bq: usize,
    scale: f32,
    scores: &mut [f32],
    bias: Option<&[f32]>,
) {
    use core::arch::x86_64::*;

    unsafe {
        let vscale = _mm512_set1_ps(scale);
        let (b0, b1, b2, b3) = match bias {
            Some(b) if tile_kv == TILE => {
                let bp = b.as_ptr().add(kv_start);
                (
                    _mm512_loadu_ps(bp),
                    _mm512_loadu_ps(bp.add(16)),
                    _mm512_loadu_ps(bp.add(32)),
                    _mm512_loadu_ps(bp.add(48)),
                )
            }
            _ => (
                _mm512_setzero_ps(),
                _mm512_setzero_ps(),
                _mm512_setzero_ps(),
                _mm512_setzero_ps(),
            ),
        };
        let add_bias_full = bias.is_some() && tile_kv == TILE;
        let add_bias_tail = bias.filter(|_| tile_kv != TILE);

        let mut qi = 0;
        // Four query rows share one K-tile walk (16 acc + 4 K loads).
        while qi + 4 <= bq && tile_kv == TILE {
            let q0 = q.as_ptr().add(qi * D);
            let q1 = q.as_ptr().add((qi + 1) * D);
            let q2 = q.as_ptr().add((qi + 2) * D);
            let q3 = q.as_ptr().add((qi + 3) * D);
            let mut a00 = _mm512_setzero_ps();
            let mut a01 = _mm512_setzero_ps();
            let mut a02 = _mm512_setzero_ps();
            let mut a03 = _mm512_setzero_ps();
            let mut a10 = _mm512_setzero_ps();
            let mut a11 = _mm512_setzero_ps();
            let mut a12 = _mm512_setzero_ps();
            let mut a13 = _mm512_setzero_ps();
            let mut a20 = _mm512_setzero_ps();
            let mut a21 = _mm512_setzero_ps();
            let mut a22 = _mm512_setzero_ps();
            let mut a23 = _mm512_setzero_ps();
            let mut a30 = _mm512_setzero_ps();
            let mut a31 = _mm512_setzero_ps();
            let mut a32 = _mm512_setzero_ps();
            let mut a33 = _mm512_setzero_ps();
            for d in 0..D {
                let kt = k_t.as_ptr().add(d * TILE);
                let k0 = _mm512_loadu_ps(kt);
                let k1 = _mm512_loadu_ps(kt.add(16));
                let k2 = _mm512_loadu_ps(kt.add(32));
                let k3 = _mm512_loadu_ps(kt.add(48));
                let qd0 = _mm512_set1_ps(*q0.add(d));
                a00 = _mm512_fmadd_ps(qd0, k0, a00);
                a01 = _mm512_fmadd_ps(qd0, k1, a01);
                a02 = _mm512_fmadd_ps(qd0, k2, a02);
                a03 = _mm512_fmadd_ps(qd0, k3, a03);
                let qd1 = _mm512_set1_ps(*q1.add(d));
                a10 = _mm512_fmadd_ps(qd1, k0, a10);
                a11 = _mm512_fmadd_ps(qd1, k1, a11);
                a12 = _mm512_fmadd_ps(qd1, k2, a12);
                a13 = _mm512_fmadd_ps(qd1, k3, a13);
                let qd2 = _mm512_set1_ps(*q2.add(d));
                a20 = _mm512_fmadd_ps(qd2, k0, a20);
                a21 = _mm512_fmadd_ps(qd2, k1, a21);
                a22 = _mm512_fmadd_ps(qd2, k2, a22);
                a23 = _mm512_fmadd_ps(qd2, k3, a23);
                let qd3 = _mm512_set1_ps(*q3.add(d));
                a30 = _mm512_fmadd_ps(qd3, k0, a30);
                a31 = _mm512_fmadd_ps(qd3, k1, a31);
                a32 = _mm512_fmadd_ps(qd3, k2, a32);
                a33 = _mm512_fmadd_ps(qd3, k3, a33);
            }
            store_qk_row4(
                scores.as_mut_ptr().add(qi * TILE),
                a00,
                a01,
                a02,
                a03,
                vscale,
                add_bias_full,
                b0,
                b1,
                b2,
                b3,
            );
            store_qk_row4(
                scores.as_mut_ptr().add((qi + 1) * TILE),
                a10,
                a11,
                a12,
                a13,
                vscale,
                add_bias_full,
                b0,
                b1,
                b2,
                b3,
            );
            store_qk_row4(
                scores.as_mut_ptr().add((qi + 2) * TILE),
                a20,
                a21,
                a22,
                a23,
                vscale,
                add_bias_full,
                b0,
                b1,
                b2,
                b3,
            );
            store_qk_row4(
                scores.as_mut_ptr().add((qi + 3) * TILE),
                a30,
                a31,
                a32,
                a33,
                vscale,
                add_bias_full,
                b0,
                b1,
                b2,
                b3,
            );
            qi += 4;
        }

        while qi < bq {
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
            if add_bias_full {
                acc0 = _mm512_add_ps(acc0, b0);
                acc1 = _mm512_add_ps(acc1, b1);
                acc2 = _mm512_add_ps(acc2, b2);
                acc3 = _mm512_add_ps(acc3, b3);
            }

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
                if let Some(b) = add_bias_tail {
                    for ki in 0..tile_kv {
                        tmp[ki] += b[kv_start + ki];
                    }
                }
                core::ptr::copy_nonoverlapping(tmp.as_ptr(), dest, tile_kv);
            }
            qi += 1;
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn store_qk_row4(
    dest: *mut f32,
    a0: core::arch::x86_64::__m512,
    a1: core::arch::x86_64::__m512,
    a2: core::arch::x86_64::__m512,
    a3: core::arch::x86_64::__m512,
    vscale: core::arch::x86_64::__m512,
    add_bias: bool,
    b0: core::arch::x86_64::__m512,
    b1: core::arch::x86_64::__m512,
    b2: core::arch::x86_64::__m512,
    b3: core::arch::x86_64::__m512,
) {
    use core::arch::x86_64::*;
    unsafe {
        let mut a0 = _mm512_mul_ps(a0, vscale);
        let mut a1 = _mm512_mul_ps(a1, vscale);
        let mut a2 = _mm512_mul_ps(a2, vscale);
        let mut a3 = _mm512_mul_ps(a3, vscale);
        if add_bias {
            a0 = _mm512_add_ps(a0, b0);
            a1 = _mm512_add_ps(a1, b1);
            a2 = _mm512_add_ps(a2, b2);
            a3 = _mm512_add_ps(a3, b3);
        }
        _mm512_storeu_ps(dest, a0);
        _mm512_storeu_ps(dest.add(16), a1);
        _mm512_storeu_ps(dest.add(32), a2);
        _mm512_storeu_ps(dest.add(48), a3);
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn pv_block_avx512(
    v: &[f32],
    kv_start: usize,
    tile_kv: usize,
    bq: usize,
    scores: &[f32],
    output: &mut [f32],
) {
    use core::arch::x86_64::*;
    unsafe {
        let mut qi = 0;
        // Four query rows share each V load. No `p == 0` skip: softmax
        // probabilities are almost never exact zero and the branch hurts.
        while qi + 4 <= bq {
            let p0 = scores.as_ptr().add(qi * TILE);
            let p1 = scores.as_ptr().add((qi + 1) * TILE);
            let p2 = scores.as_ptr().add((qi + 2) * TILE);
            let p3 = scores.as_ptr().add((qi + 3) * TILE);
            let o0 = output.as_mut_ptr().add(qi * D);
            let o1 = output.as_mut_ptr().add((qi + 1) * D);
            let o2 = output.as_mut_ptr().add((qi + 2) * D);
            let o3 = output.as_mut_ptr().add((qi + 3) * D);
            let mut a00 = _mm512_loadu_ps(o0);
            let mut a01 = _mm512_loadu_ps(o0.add(16));
            let mut a10 = _mm512_loadu_ps(o1);
            let mut a11 = _mm512_loadu_ps(o1.add(16));
            let mut a20 = _mm512_loadu_ps(o2);
            let mut a21 = _mm512_loadu_ps(o2.add(16));
            let mut a30 = _mm512_loadu_ps(o3);
            let mut a31 = _mm512_loadu_ps(o3.add(16));
            for ki in 0..tile_kv {
                let vptr = v.as_ptr().add((kv_start + ki) * D);
                let v0 = _mm512_loadu_ps(vptr);
                let v1 = _mm512_loadu_ps(vptr.add(16));
                let s0 = _mm512_set1_ps(*p0.add(ki));
                a00 = _mm512_fmadd_ps(s0, v0, a00);
                a01 = _mm512_fmadd_ps(s0, v1, a01);
                let s1 = _mm512_set1_ps(*p1.add(ki));
                a10 = _mm512_fmadd_ps(s1, v0, a10);
                a11 = _mm512_fmadd_ps(s1, v1, a11);
                let s2 = _mm512_set1_ps(*p2.add(ki));
                a20 = _mm512_fmadd_ps(s2, v0, a20);
                a21 = _mm512_fmadd_ps(s2, v1, a21);
                let s3 = _mm512_set1_ps(*p3.add(ki));
                a30 = _mm512_fmadd_ps(s3, v0, a30);
                a31 = _mm512_fmadd_ps(s3, v1, a31);
            }
            _mm512_storeu_ps(o0, a00);
            _mm512_storeu_ps(o0.add(16), a01);
            _mm512_storeu_ps(o1, a10);
            _mm512_storeu_ps(o1.add(16), a11);
            _mm512_storeu_ps(o2, a20);
            _mm512_storeu_ps(o2.add(16), a21);
            _mm512_storeu_ps(o3, a30);
            _mm512_storeu_ps(o3.add(16), a31);
            qi += 4;
        }
        while qi < bq {
            let prow = scores.as_ptr().add(qi * TILE);
            let optr = output.as_mut_ptr().add(qi * D);
            let mut acc0 = _mm512_loadu_ps(optr);
            let mut acc1 = _mm512_loadu_ps(optr.add(16));
            for ki in 0..tile_kv {
                let pv = _mm512_set1_ps(*prow.add(ki));
                let vptr = v.as_ptr().add((kv_start + ki) * D);
                acc0 = _mm512_fmadd_ps(pv, _mm512_loadu_ps(vptr), acc0);
                acc1 = _mm512_fmadd_ps(pv, _mm512_loadu_ps(vptr.add(16)), acc1);
            }
            _mm512_storeu_ps(optr, acc0);
            _mm512_storeu_ps(optr.add(16), acc1);
            qi += 1;
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

    fn make_e5_like(
        batch: usize,
        heads: usize,
        seq: usize,
    ) -> (FlexTensor, FlexTensor, FlexTensor) {
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
    fn transpose_k_tile_matches_scalar() {
        let k: Vec<f32> = (0..2 * TILE * D)
            .map(|i| (i as f32) * 0.017 - 3.1)
            .collect();
        let mut simd = [0.0f32; D * TILE];
        let mut scalar = [0.0f32; D * TILE];
        for ki in 0..TILE {
            let krow = &k[ki * D..];
            for d in 0..D {
                scalar[d * TILE + ki] = krow[d];
            }
        }
        transpose_k_tile(&k, 0, TILE, &mut simd);
        for i in 0..D * TILE {
            assert_eq!(
                simd[i], scalar[i],
                "k_t[{i}] simd={} scalar={}",
                simd[i], scalar[i]
            );
        }
        // Offset tile (kv_start = TILE) and a short tail.
        transpose_k_tile(&k, TILE, TILE, &mut simd);
        for ki in 0..TILE {
            let krow = &k[(TILE + ki) * D..];
            for d in 0..D {
                assert_eq!(simd[d * TILE + ki], krow[d]);
            }
        }
        transpose_k_tile(&k, 0, 17, &mut simd);
        for d in 0..D {
            for ki in 0..TILE {
                let expect = if ki < 17 { k[ki * D + d] } else { 0.0 };
                assert_eq!(simd[d * TILE + ki], expect);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn online_softmax_4_matches_one() {
        if !avx512f_available() {
            return;
        }
        let dummy = HeadParams {
            seq_q: 4,
            seq_kv: TILE,
            scale: 1.0,
            causal_offset: None,
            q_batch_stride: 0,
            k_batch_stride: 0,
            v_batch_stride: 0,
            q_head_stride: 0,
            k_head_stride: 0,
            v_head_stride: 0,
            q_per_kv: 1,
            mask_batch_step: 0,
            mask_head_step: 0,
            mask_q_step: 0,
            mask_tile_len: 0,
            bias_batch_step: 0,
            bias_head_step: 0,
            bias_q_step: 0,
            bias_tile_len: 0,
        };
        let check = |mask_row1: bool| {
            let mut four = [0.0f32; 4 * TILE];
            let mut one = [0.0f32; 4 * TILE];
            for i in 0..4 * TILE {
                let v = ((i % 53) as f32) * 0.11 - 2.4;
                four[i] = v;
                one[i] = v;
            }
            if mask_row1 {
                for x in &mut four[TILE..2 * TILE] {
                    *x = f32::NEG_INFINITY;
                }
                for x in &mut one[TILE..2 * TILE] {
                    *x = f32::NEG_INFINITY;
                }
            }
            let mut max4 = [f32::NEG_INFINITY; 4];
            let mut max1 = [f32::NEG_INFINITY; 4];
            let mut sum4 = [0.0f32; 4];
            let mut sum1 = [0.0f32; 4];
            let mut out4 = [0.3f32; 4 * D];
            let mut out1 = [0.3f32; 4 * D];
            // SAFETY: gated on avx512f; four full TILE rows.
            unsafe {
                online_softmax_4(&mut four, 0, &mut max4, &mut sum4, &mut out4);
            }
            for qi in 0..4 {
                online_softmax_one(
                    &mut one[qi * TILE..(qi + 1) * TILE],
                    qi,
                    &mut max1,
                    &mut sum1,
                    &mut out1,
                    None,
                    None,
                    None,
                    0,
                    &dummy,
                );
            }
            for i in 0..4 * TILE {
                let err = (four[i] - one[i]).abs();
                assert!(
                    err < 1e-6,
                    "mask_row1={mask_row1} scores[{i}] 4-row={} one={} err={err:e}",
                    four[i],
                    one[i]
                );
            }
            for qi in 0..4 {
                assert!((max4[qi] - max1[qi]).abs() < 1e-6 || max4[qi] == max1[qi]);
                assert!((sum4[qi] - sum1[qi]).abs() < 1e-5);
                for d in 0..D {
                    let a = out4[qi * D + d];
                    let b = out1[qi * D + d];
                    assert!(
                        (a - b).abs() < 1e-6,
                        "mask_row1={mask_row1} out[{qi},{d}] {a} vs {b}"
                    );
                }
            }
        };
        check(false);
        check(true);
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
            (0..seq)
                .map(|i| if i < 10 { -1.0e4 } else { 0.0 })
                .collect(),
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
