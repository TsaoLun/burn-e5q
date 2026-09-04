//! Compare burn int8 E5 embeddings against ort reference vectors.
//!
//! 1. Generate reference data:  `uv run scripts/gen_ref.py`
//! 2. Run:                      `cargo run --release -p e5-embed --bin compare_ort`
//!
//! Checks tokenizer parity (sentencepiece vs HF tokenizers ids), embedding
//! cosine similarity per case, and measures burn latency at several batch
//! shapes against the ort baseline recorded in the reference file.

use std::time::Instant;

use anyhow::Context;
use serde::Deserialize;

use e5_embed::{
    E5_PASSAGE_PREFIX, E5_QUERY_PREFIX, E5Embedder, current_rss_hwm_mb, default_model_dir,
};

#[derive(Debug, Deserialize)]
struct RefCase {
    prefix: String,
    text: String,
    ids: Vec<i64>,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct RefLatency {
    ort_single_ms: f64,
    ort_batch8_ms: f64,
    ort_long512_ms: f64,
}

#[derive(Debug, Deserialize)]
struct RefData {
    model: String,
    cases: Vec<RefCase>,
    latency: RefLatency,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn capped_tokens(embedder: &E5Embedder, prefix: &str, text: &str) -> usize {
    embedder
        .encode_prefixed(prefix, text)
        .map(|ids| ids.len().min(512))
        .unwrap_or(0)
}

fn print_perf_row(label: &str, burn_ms: f64, ort_ms: f64, n_q: usize, n_tok: usize) {
    let qps = n_q as f64 * 1e3 / burn_ms.max(1e-9);
    let tps = n_tok as f64 * 1e3 / burn_ms.max(1e-9);
    let ort_qps = n_q as f64 * 1e3 / ort_ms.max(1e-9);
    println!(
        "  {label}: burn {burn_ms:8.1} ms ({qps:6.2} q/s, {tps:7.0} tok/s) | ort {ort_ms:6.1} ms ({ort_qps:6.2} q/s) | {ratio:.1}×",
        ratio = burn_ms / ort_ms.max(1e-9)
    );
}

fn rank_top3(corpus: &[&RefCase], query_emb: &[f32]) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, c)| (i, cosine(&c.embedding, query_emb)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.iter().take(3).map(|(i, _)| *i).collect()
}

fn rank_top3_burn(burn_passages: &[Vec<f32>], query_emb: &[f32]) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = burn_passages
        .iter()
        .enumerate()
        .map(|(i, p)| (i, cosine(p, query_emb)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.iter().take(3).map(|(i, _)| *i).collect()
}

fn paired_queries<'a>(
    ref_data: &'a RefData,
    embedder: &E5Embedder,
) -> anyhow::Result<Vec<(&'a RefCase, Vec<f32>)>> {
    let query_cases: Vec<&RefCase> = ref_data
        .cases
        .iter()
        .filter(|c| c.prefix == E5_QUERY_PREFIX)
        .collect();
    let got = embedder.embed_queries(
        &query_cases
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>(),
    )?;
    Ok(query_cases.into_iter().zip(got).collect())
}

fn main() -> anyhow::Result<()> {
    let ref_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/ref_data.json").to_string());
    let ref_data: RefData = serde_json::from_slice(
        &std::fs::read(&ref_path).with_context(|| format!("read {ref_path}"))?,
    )
    .context("parse reference data")?;
    println!("Reference model: {}", ref_data.model);
    println!("Reference file:  {ref_path}\n");

    let (rss, hwm) = current_rss_hwm_mb();
    match hwm {
        Some(hwm) => println!("RSS before model load: {rss:.1} MB  HWM {hwm:.1} MB"),
        None => println!("RSS before model load: {rss:.1} MB"),
    }
    let load_start = Instant::now();
    let device = burn::prelude::Device::default();
    println!(
        "Device: {device:?} ({})",
        if cfg!(feature = "cpu") {
            "cubecl-cpu"
        } else {
            "flex"
        }
    );
    let embedder = E5Embedder::load(&default_model_dir(), &device)?;
    let (rss, hwm) = current_rss_hwm_mb();
    match hwm {
        Some(hwm) => println!(
            "Model loaded in {:.2?}. RSS: {rss:.1} MB  HWM {hwm:.1} MB",
            load_start.elapsed()
        ),
        None => println!(
            "Model loaded in {:.2?}. RSS: {rss:.1} MB",
            load_start.elapsed()
        ),
    }

    // 1. Tokenizer parity: sentencepiece (burn side) vs HF tokenizers ids.
    println!("\n=== Tokenizer parity ===");
    let mut tokenizer_mismatch = 0usize;
    for case in &ref_data.cases {
        if case.text.trim().is_empty() {
            continue;
        }
        let ids = embedder.encode_prefixed(&case.prefix, &case.text)?;
        if ids != case.ids {
            tokenizer_mismatch += 1;
            println!(
                "  ✗ id mismatch for {:?}{:?}\n    burn: {:?}\n    ort:  {:?}",
                case.prefix,
                case.text,
                &ids[..ids.len().min(24)],
                &case.ids[..case.ids.len().min(24)]
            );
        }
    }
    if tokenizer_mismatch == 0 {
        println!(
            "  ✓ all {} non-empty cases have identical ids",
            ref_data
                .cases
                .iter()
                .filter(|c| !c.text.trim().is_empty())
                .count()
        );
    } else {
        println!("  ✗ {tokenizer_mismatch} cases mismatch");
    }

    // 2. Embedding cosine similarity (full pipeline: prefix → tokenize →
    //    budget-packed forward → mean pool → L2 normalize).
    println!("\n=== Embedding cosine similarity (burn vs ort) ===");
    let passage_cases: Vec<&RefCase> = ref_data
        .cases
        .iter()
        .filter(|c| c.prefix == E5_PASSAGE_PREFIX)
        .collect();
    let query_cases: Vec<&RefCase> = ref_data
        .cases
        .iter()
        .filter(|c| c.prefix == E5_QUERY_PREFIX)
        .collect();
    let got_passages = embedder.embed_passages(
        &passage_cases
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>(),
    )?;
    let got_queries = embedder.embed_queries(
        &query_cases
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>(),
    )?;
    let paired = passage_cases
        .iter()
        .zip(&got_passages)
        .chain(query_cases.iter().zip(&got_queries));

    let mut min_cos = f32::INFINITY;
    let mut sum_cos = 0.0f64;
    for (case, burn_emb) in paired {
        if case.text.trim().is_empty() {
            let is_zero = burn_emb.iter().all(|&x| x == 0.0);
            println!(
                "  empty text -> zero vector: {}",
                if is_zero { "✓" } else { "✗" }
            );
            continue;
        }
        let cos = cosine(burn_emb, &case.embedding);
        min_cos = min_cos.min(cos);
        sum_cos += cos as f64;
        let mark = if cos > 0.999 { "✓" } else { "✗" };
        let label: String = case.text.chars().take(28).collect();
        println!(
            "  {mark} cos={cos:.6}  {}{:?}",
            case.prefix.trim_end_matches(": "),
            label
        );
    }
    let n = ref_data
        .cases
        .iter()
        .filter(|c| !c.text.trim().is_empty())
        .count();
    println!(
        "  min cos = {min_cos:.6}, mean cos = {:.6} over {n} cases",
        sum_cos / n as f64
    );

    // 2b. Retrieval ranking parity: cos values below 1.0 are expected for
    // int8 (rounding-boundary divergence is documented inside ORT itself
    // across platforms); what matters for retrieval is identical ranking.
    // Rank all passages by each query and compare top-k order.
    println!("\n=== Retrieval ranking parity (top-3) ===");
    let corpus: Vec<&RefCase> = ref_data
        .cases
        .iter()
        .filter(|c| c.prefix == E5_PASSAGE_PREFIX && !c.text.trim().is_empty())
        .collect();
    let burn_passages =
        embedder.embed_passages(&corpus.iter().map(|c| c.text.as_str()).collect::<Vec<_>>())?;
    let mut rank_match = 0usize;
    let mut rank_total = 0usize;
    for (case, burn_q) in paired_queries(&ref_data, &embedder)? {
        let ort_ranking = rank_top3(&corpus, &case.embedding);
        let burn_ranking = rank_top3_burn(&burn_passages, &burn_q);
        let same = ort_ranking == burn_ranking;
        rank_match += usize::from(same);
        rank_total += 1;
        let label: String = case.text.chars().take(24).collect();
        println!(
            "  {} query {:?}: ort {:?} vs burn {:?}",
            if same { "✓" } else { "✗" },
            label,
            ort_ranking,
            burn_ranking
        );
    }
    println!("  ranking match: {rank_match}/{rank_total}");

    // 3. Latency: single short passage, all-at-once batch, one 512-token row.
    println!(
        "\n=== Latency ({}, release) ===",
        if cfg!(feature = "cpu") {
            "cubecl-cpu"
        } else {
            "flex"
        }
    );
    let single = ["周末滨江夜骑 V11，速度很快！"];
    let single_toks = capped_tokens(&embedder, E5_PASSAGE_PREFIX, single[0]);
    let mut burn_single = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let _ = embedder.embed_passages(&single)?;
        burn_single = burn_single.min(t.elapsed().as_secs_f64() * 1e3);
    }
    print_perf_row(
        "single short passage ",
        burn_single,
        ref_data.latency.ort_single_ms,
        1,
        single_toks,
    );

    let passage_texts: Vec<&str> = passage_cases.iter().map(|c| c.text.as_str()).collect();
    let batch_toks: usize = passage_texts
        .iter()
        .map(|t| capped_tokens(&embedder, E5_PASSAGE_PREFIX, t))
        .sum();
    let t = Instant::now();
    let _ = embedder.embed_passages(&passage_texts)?;
    let burn_batch = t.elapsed().as_secs_f64() * 1e3;
    print_perf_row(
        &format!("batch of {}          ", passage_texts.len()),
        burn_batch,
        ref_data.latency.ort_batch8_ms,
        passage_texts.len(),
        batch_toks,
    );

    let long_text = ref_data
        .cases
        .iter()
        .map(|c| c.text.as_str())
        .max_by_key(|t| t.len())
        .unwrap_or("night ride");
    let long_toks = capped_tokens(&embedder, E5_PASSAGE_PREFIX, long_text);
    let t = Instant::now();
    let _ = embedder.embed_passages(&[long_text])?;
    let burn_long = t.elapsed().as_secs_f64() * 1e3;
    print_perf_row(
        "single long (512 tok)",
        burn_long,
        ref_data.latency.ort_long512_ms,
        1,
        long_toks,
    );

    // embed_passages includes sentencepiece. The Rust ort 54 ms baseline is
    // session.run on pre-encoded ids — split so we stop comparing those.
    let mut tok_long = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let _ = embedder.encode_prefixed(E5_PASSAGE_PREFIX, long_text)?;
        tok_long = tok_long.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let long_ids = embedder.encode_prefixed(E5_PASSAGE_PREFIX, long_text)?;
    let seq = long_ids.len().min(512);
    let mut input = vec![1i64; seq];
    let mut mask = vec![0i64; seq];
    input[..seq].copy_from_slice(&long_ids[..seq]);
    mask.fill(1);
    use burn::prelude::*;
    let ids_t = Tensor::<2, Int>::from_data(TensorData::new(input, [1, seq]), &device);
    let mask_t = Tensor::<2, Int>::from_data(TensorData::new(mask, [1, seq]), &device);
    let tt_t = Tensor::<2, Int>::zeros([1, seq], &device);
    let mut fwd_long = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let h = embedder.forward_raw(ids_t.clone(), mask_t.clone(), tt_t.clone());
        let _ = std::hint::black_box(h);
        fwd_long = fwd_long.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "  long split: tokenize {tok_long:.1} ms + forward_raw {fwd_long:.1} ms (embed_passages {burn_long:.1})"
    );

    // Same 16-token short passage, model only. Rust ort short is session.run
    // on pre-encoded ids (`ort-mem`); embed_passages includes sentencepiece.
    let short_ids = embedder.encode_prefixed(E5_PASSAGE_PREFIX, single[0])?;
    let short_seq = short_ids.len().min(512);
    let mut short_input = vec![1i64; short_seq];
    let mut short_mask = vec![0i64; short_seq];
    short_input[..short_seq].copy_from_slice(&short_ids[..short_seq]);
    short_mask.fill(1);
    let short_ids_t =
        Tensor::<2, Int>::from_data(TensorData::new(short_input, [1, short_seq]), &device);
    let short_mask_t =
        Tensor::<2, Int>::from_data(TensorData::new(short_mask, [1, short_seq]), &device);
    let short_tt_t = Tensor::<2, Int>::zeros([1, short_seq], &device);
    let mut tok_short = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let _ = embedder.encode_prefixed(E5_PASSAGE_PREFIX, single[0])?;
        tok_short = tok_short.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut fwd_short = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let h = embedder.forward_raw(short_ids_t.clone(), short_mask_t.clone(), short_tt_t.clone());
        let _ = std::hint::black_box(h);
        fwd_short = fwd_short.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "  short split: tokenize {tok_short:.1} ms + forward_raw {fwd_short:.1} ms (embed_passages {burn_single:.1})"
    );

    // Live `cargo run --release -p ort-mem` on this 4-core Xeon (2026-09-04):
    // arena off, pre-encoded ids, session.run + mean-pool. Not the Mac
    // Python numbers in ref_data.json (4.3 / 1412 / 201).
    const RUST_ORT_SHORT_MS: f64 = 2.4;
    const RUST_ORT_PACKED_MS: f64 = 923.4;
    const RUST_ORT_512_MS: f64 = 39.3;
    println!("\n=== vs this-machine Rust ort (ort-mem, arena off) ===");
    println!(
        "  short model:   forward_raw {fwd_short:.1} / {RUST_ORT_SHORT_MS:.1} = {:.1}×  (embed_passages {burn_single:.1} = {:.1}×, includes SP)",
        fwd_short / RUST_ORT_SHORT_MS,
        burn_single / RUST_ORT_SHORT_MS
    );
    println!(
        "  packed batch:  embed_passages {burn_batch:.1} / {RUST_ORT_PACKED_MS:.1} = {:.1}×  (burn includes SP; ort is session only)",
        burn_batch / RUST_ORT_PACKED_MS
    );
    println!(
        "  512 model:     forward_raw {fwd_long:.1} / {RUST_ORT_512_MS:.1} = {:.1}×  (embed_passages {burn_long:.1} = {:.1}×, includes SP)",
        fwd_long / RUST_ORT_512_MS,
        burn_long / RUST_ORT_512_MS
    );

    println!("\n=== Throughput ===");
    println!(
        "  short : {:>7.2} q/s   {:>8.0} tok/s   ({} tok)",
        1e3 / burn_single.max(1e-9),
        single_toks as f64 * 1e3 / burn_single.max(1e-9),
        single_toks
    );
    println!(
        "  batch : {:>7.2} q/s   {:>8.0} tok/s   ({} tok / {} q)",
        passage_texts.len() as f64 * 1e3 / burn_batch.max(1e-9),
        batch_toks as f64 * 1e3 / burn_batch.max(1e-9),
        batch_toks,
        passage_texts.len()
    );
    println!(
        "  512   : {:>7.2} q/s   {:>8.0} tok/s   ({} tok)",
        1e3 / burn_long.max(1e-9),
        long_toks as f64 * 1e3 / burn_long.max(1e-9),
        long_toks
    );

    let (rss, hwm) = current_rss_hwm_mb();
    match hwm {
        Some(hwm) => {
            println!("\nRSS after all inference: {rss:.1} MB");
            println!("kernel peak HWM:         {hwm:.1} MB");
        }
        None => println!("\nRSS after all inference: {rss:.1} MB"),
    }
    Ok(())
}
