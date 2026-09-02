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
    E5_PASSAGE_PREFIX, E5_QUERY_PREFIX, E5Embedder, current_hwm_mb, current_rss_mb, default_model_dir,
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
        &query_cases.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
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

    match current_hwm_mb() {
        Some(hwm) => println!(
            "RSS before model load: {:.1} MB  HWM {:.1} MB",
            current_rss_mb(),
            hwm
        ),
        None => println!("RSS before model load: {:.1} MB", current_rss_mb()),
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
    match current_hwm_mb() {
        Some(hwm) => println!(
            "Model loaded in {:.2?}. RSS: {:.1} MB  HWM {:.1} MB",
            load_start.elapsed(),
            current_rss_mb(),
            hwm
        ),
        None => println!(
            "Model loaded in {:.2?}. RSS: {:.1} MB",
            load_start.elapsed(),
            current_rss_mb()
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
        println!("  ✓ all {} non-empty cases have identical ids", ref_data.cases.iter().filter(|c| !c.text.trim().is_empty()).count());
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
        &passage_cases.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
    )?;
    let got_queries = embedder.embed_queries(
        &query_cases.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
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
            println!("  empty text -> zero vector: {}", if is_zero { "✓" } else { "✗" });
            continue;
        }
        let cos = cosine(burn_emb, &case.embedding);
        min_cos = min_cos.min(cos);
        sum_cos += cos as f64;
        let mark = if cos > 0.999 { "✓" } else { "✗" };
        let label: String = case.text.chars().take(28).collect();
        println!("  {mark} cos={cos:.6}  {}{:?}", case.prefix.trim_end_matches(": "), label);
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
    let burn_passages = embedder.embed_passages(
        &corpus.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
    )?;
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

    println!("\nRSS after all inference: {:.1} MB", current_rss_mb());
    if let Some(hwm) = current_hwm_mb() {
        println!("kernel peak HWM:         {hwm:.1} MB");
    }
    Ok(())
}
