//! Integer QK for long f32 flash attention (route C-lite).
//!
//! e5 512×512 flash spends most of its time in tiny-K f32 gemm (~12 GOPS).
//! This path keeps the public `attention()` f32 API and, for long sequences:
//!
//! 1. Per-head ONNX-style DynamicQuantizeLinear on Q and K (`u8`)
//! 2. One VNNI `u8×u8 → i32` GEMM for `Q @ K^T` (K is transposed to `[D,S]`)
//! 3. Dequant × attention scale, mask / bias, row softmax in f32
//! 4. `P @ V` in f32 (AVX-512 when `val_dim == 32`)
//!
//! One head at a time, so `[H,S,S]` is never materialized. V stays f32 — a
//! second quantize on softmax/P would add drift without shrinking the exp
//! loop. Softcap is unsupported.
//!
//! Gated to `seq_q, seq_kv ≥ 256` so existing flash unit tests stay on the
//! bit-exact f32 gemm path.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::DType;
use burn_backend::ops::AttentionModuleOptions;
use burn_std::Bytes;

use crate::ops::attention::broadcast_attn_mask_bias;
use crate::ops::dql::dql_u8;
use crate::ops::int_gemm::{self, Zp, ZpLane};
use crate::{FlexTensor, Layout};

/// Minimum sequence on both Q and KV before this path may fire.
const MIN_SEQ: usize = 256;

#[allow(dead_code)] // kept for tests; not hooked into attention_flash (slower on e5)
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
    let seq_q = q[2];
    let seq_kv = k[2];
    let head_dim = q[3];
    if seq_q < MIN_SEQ || seq_kv < MIN_SEQ {
        return false;
    }
    if head_dim < 16 || !head_dim.is_multiple_of(4) {
        return false;
    }
    int_gemm::vnni_available()
}

#[allow(dead_code)] // see should_use
pub(crate) fn flash_int8(
    query: FlexTensor,
    key: FlexTensor,
    value: FlexTensor,
    mask: Option<FlexTensor>,
    attn_bias: Option<FlexTensor>,
    options: AttentionModuleOptions,
) -> FlexTensor {
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

    let q_head_stride = seq_q * head_dim;
    let q_batch_stride = heads * q_head_stride;
    let k_head_stride = seq_kv * head_dim;
    let k_batch_stride = kv_heads * k_head_stride;
    let v_head_stride = seq_kv * val_dim;
    let v_batch_stride = kv_heads * v_head_stride;
    let o_head_stride = seq_q * val_dim;

    let mut output = vec![0.0f32; batch * heads * seq_q * val_dim];

    let params = HeadParams {
        scale,
        causal_offset,
        seq_q,
        seq_kv,
        head_dim,
        val_dim,
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
        use rayon::prelude::*;
        output
            .par_chunks_mut(o_head_stride)
            .enumerate()
            .for_each(|(idx, out_head)| {
                let b = idx / heads;
                let h = idx % heads;
                one_head(
                    b, h, q_data, k_data, v_data, out_head, mask_data, bias_data, &params,
                );
            });
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

    let shape = burn_std::Shape::from(vec![batch, heads, seq_q, val_dim]);
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
    head_dim: usize,
    val_dim: usize,
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

    int8_qk_head(
        &q_data[q_off..q_off + p.q_head_stride],
        &k_data[k_off..k_off + p.k_head_stride],
        &v_data[v_off..v_off + p.v_head_stride],
        out_head,
        mask_data.map(|m| &m[mask_off..mask_off + p.mask_tile_len]),
        bias_data.map(|bias| &bias[bias_off..bias_off + p.bias_tile_len]),
        p,
    );
}

fn int8_qk_head(
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
    let head_dim = p.head_dim;
    let val_dim = p.val_dim;

    let (q_u8, q_scale, q_zp) = dql_u8(q);
    let (k_u8, k_scale, k_zp) = dql_u8(k);
    let score_scale = q_scale * k_scale * p.scale;

    // K[seq_kv, D] → B[D, seq_kv] for VNNI A[m,k] @ B[k,n].
    let mut k_t = vec![0u8; head_dim * seq_kv];
    transpose_u8(&k_u8, seq_kv, head_dim, &mut k_t);

    let zp = Zp {
        za: ZpLane::Scalar(q_zp as i32),
        zb: ZpLane::Scalar(k_zp as i32),
    };
    let mut scores_i32 = vec![0i32; seq_q * seq_kv];
    int_gemm::gemm_u8u8_into(&q_u8, &k_t, seq_q, seq_kv, head_dim, &zp, &mut scores_i32);

    let mut scores = vec![0.0f32; seq_q * seq_kv];
    dequant_scale_mask_softmax(
        &scores_i32,
        &mut scores,
        score_scale,
        mask,
        bias,
        p.causal_offset,
        seq_q,
        seq_kv,
        p.mask_q_step,
        p.bias_q_step,
    );

    output.fill(0.0);
    pv_f32(&scores, v, output, seq_q, seq_kv, val_dim);
}

fn transpose_u8(src: &[u8], rows: usize, cols: usize, dst: &mut [u8]) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);
    for r in 0..rows {
        let src_row = &src[r * cols..r * cols + cols];
        for c in 0..cols {
            dst[c * rows + r] = src_row[c];
        }
    }
}

fn dequant_scale_mask_softmax(
    scores_i32: &[i32],
    scores: &mut [f32],
    score_scale: f32,
    mask: Option<&[u8]>,
    bias: Option<&[f32]>,
    causal_offset: Option<isize>,
    seq_q: usize,
    seq_kv: usize,
    mask_q_step: usize,
    bias_q_step: usize,
) {
    let neg_inf = f32::NEG_INFINITY;
    for qi in 0..seq_q {
        let in_row = &scores_i32[qi * seq_kv..qi * seq_kv + seq_kv];
        let out_row = &mut scores[qi * seq_kv..qi * seq_kv + seq_kv];
        let mut row_max = neg_inf;
        for ki in 0..seq_kv {
            let mut val = in_row[ki] as f32 * score_scale;
            if let Some(m) = mask
                && m[qi * mask_q_step + ki] != 0
            {
                val = neg_inf;
            }
            if let Some(offset) = causal_offset
                && (ki as isize) > (qi as isize) + offset
            {
                val = neg_inf;
            }
            if let Some(b) = bias {
                val += b[qi * bias_q_step + ki];
            }
            out_row[ki] = val;
            if val > row_max {
                row_max = val;
            }
        }
        if row_max == neg_inf {
            out_row.fill(0.0);
            continue;
        }
        let mut sum = 0.0f32;
        for s in out_row.iter_mut() {
            let e = (*s - row_max).exp();
            *s = e;
            sum += e;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for s in out_row.iter_mut() {
                *s *= inv;
            }
        }
    }
}

fn pv_f32(p: &[f32], v: &[f32], out: &mut [f32], seq_q: usize, seq_kv: usize, val_dim: usize) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if val_dim == 32 && is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f checked.
            unsafe {
                pv_f32_avx512_d32(p, v, out, seq_q, seq_kv);
            }
            return;
        }
    }
    for qi in 0..seq_q {
        let p_row = &p[qi * seq_kv..qi * seq_kv + seq_kv];
        let o_row = &mut out[qi * val_dim..qi * val_dim + val_dim];
        o_row.fill(0.0);
        for ki in 0..seq_kv {
            let pk = p_row[ki];
            if pk == 0.0 {
                continue;
            }
            let v_row = &v[ki * val_dim..ki * val_dim + val_dim];
            for d in 0..val_dim {
                o_row[d] += pk * v_row[d];
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx512f")]
unsafe fn pv_f32_avx512_d32(p: &[f32], v: &[f32], out: &mut [f32], seq_q: usize, seq_kv: usize) {
    use core::arch::x86_64::*;
    const D: usize = 32;
    for qi in 0..seq_q {
        let p_row = &p[qi * seq_kv..qi * seq_kv + seq_kv];
        unsafe {
            let o_ptr = out.as_mut_ptr().add(qi * D);
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            for ki in 0..seq_kv {
                let pk = p_row[ki];
                if pk == 0.0 {
                    continue;
                }
                let pv = _mm512_set1_ps(pk);
                let v_ptr = v.as_ptr().add(ki * D);
                acc0 = _mm512_fmadd_ps(pv, _mm512_loadu_ps(v_ptr), acc0);
                acc1 = _mm512_fmadd_ps(pv, _mm512_loadu_ps(v_ptr.add(16)), acc1);
            }
            _mm512_storeu_ps(o_ptr, acc0);
            _mm512_storeu_ps(o_ptr.add(16), acc1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::attention::attention_naive;
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

    #[test]
    fn int8_flash_stays_off_for_short_seq() {
        let q = flex_f32(vec![1.0; 1 * 1 * 8 * 32], &[1, 1, 8, 32]);
        let k = q.clone();
        let v = q.clone();
        assert!(!should_use(&q, &k, &v, &Default::default()));
    }

    #[test]
    fn int8_flash_close_to_f32_naive_e5_like() {
        if !int_gemm::vnni_available() {
            return;
        }
        let batch = 1;
        let heads = 2;
        let seq = 256;
        let dim = 32;
        let n = batch * heads * seq * dim;
        let q = flex_f32(
            (0..n).map(|i| ((i % 97) as f32) * 0.02 - 0.7).collect(),
            &[batch, heads, seq, dim],
        );
        let k = flex_f32(
            (0..n).map(|i| ((i % 89) as f32) * 0.017 - 0.4).collect(),
            &[batch, heads, seq, dim],
        );
        let v = flex_f32(
            (0..n).map(|i| ((i % 83) as f32) * 0.013 - 0.2).collect(),
            &[batch, heads, seq, dim],
        );
        assert!(should_use(&q, &k, &v, &Default::default()));

        let int8 = flash_int8(
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
        let ia: &[f32] = int8.storage();
        let na: &[f32] = naive.storage();
        let cos = cosine(ia, na);
        assert!(
            cos > 0.98,
            "int8 flash vs f32 naive cosine {cos} (expected > 0.98)"
        );
    }
}
