//! Diagnose where burn and ort diverge inside the int8 graph.
//!
//! Compares the raw `last_hidden_state` (before mean pooling) of the first
//! reference case element-by-element and prints the difference distribution.
//!
//! Run: `uv run scripts/gen_ref.py` first, then
//!      `cargo run --release -p e5-embed --bin diagnose`

use anyhow::Context;
use serde::Deserialize;

use e5_embed::{E5Embedder, default_model_dir};

#[derive(Debug, Deserialize)]
struct RefCase {
    prefix: String,
    text: String,
    ids: Vec<i64>,
    embedding: Vec<f32>,
    last_hidden: Option<Vec<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
struct RefData {
    cases: Vec<RefCase>,
}

fn main() -> anyhow::Result<()> {
    let ref_path = concat!(env!("CARGO_MANIFEST_DIR"), "/ref_data.json");
    let ref_data: RefData = serde_json::from_slice(
        &std::fs::read(ref_path).with_context(|| format!("read {ref_path}"))?,
    )?;
    let case = ref_data
        .cases
        .iter()
        .find(|c| c.last_hidden.is_some())
        .context("no case with last_hidden; re-run gen_ref.py")?;
    let ref_hidden = case.last_hidden.as_ref().unwrap();
    let seq_len = ref_hidden.len();
    println!(
        "Case: {:?}{:?} ({} tokens, hidden {})",
        case.prefix,
        case.text,
        seq_len,
        ref_hidden[0].len()
    );

    let device = burn::prelude::Device::default();
    let embedder = E5Embedder::load(&default_model_dir(), &device)?;

    // Replay the exact ort-padded input through the burn model so the graphs
    // see identical inputs (including pad tokens), then compare raw outputs.
    let ids = &case.ids;
    let batch = 1usize;
    let mut input_ids = vec![1i64; batch * seq_len];
    let mut attention = vec![0i64; batch * seq_len];
    input_ids[..seq_len].copy_from_slice(ids);
    attention[..seq_len].fill(1);
    let _ = &attention; // padding-free here; mask only matters for pooling

    use burn::prelude::*;
    let shape = [batch, seq_len];
    let input_ids_t = Tensor::<2, Int>::from_data(TensorData::new(input_ids, shape), &device);
    let attention_t = Tensor::<2, Int>::from_data(TensorData::new(attention, shape), &device);
    let token_type_t = Tensor::<2, Int>::zeros(shape, &device);

    let hidden = embedder.forward_raw(input_ids_t, attention_t, token_type_t);
    let burn_flat: Vec<f32> = hidden.into_data().try_to_vec::<f32>().expect("f32");

    let ref_flat: Vec<f32> = ref_hidden.iter().flatten().copied().collect();
    assert_eq!(burn_flat.len(), ref_flat.len(), "shape mismatch");

    let mut diffs: Vec<(usize, f32, f32, f32)> = burn_flat
        .iter()
        .zip(&ref_flat)
        .enumerate()
        .map(|(i, (&b, &r))| (i, b, r, (b - r).abs()))
        .collect();
    let max_diff = diffs.iter().map(|d| d.3).fold(0.0f32, f32::max);
    let mean_diff = diffs.iter().map(|d| d.3).sum::<f32>() / diffs.len() as f32;
    let ref_abs_max = ref_flat.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    println!(
        "elements: {}, ref |max|: {:.4}",
        diffs.len(),
        ref_abs_max
    );
    println!("abs diff: max {max_diff:.6}, mean {mean_diff:.6}");

    let buckets = [1e-6f32, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, f32::INFINITY];
    let mut counts = [0usize; 7];
    for d in &diffs {
        let idx = buckets.iter().position(|&b| d.3 < b).unwrap();
        counts[idx] += 1;
    }
    let labels = ["<1e-6", "1e-6..1e-5", "1e-5..1e-4", "1e-4..1e-3", "1e-3..1e-2", "1e-2..1e-1", ">=1e-1"];
    for (label, count) in labels.iter().zip(&counts) {
        let pct = *count as f64 / diffs.len() as f64 * 100.0;
        println!("  {label:>12}: {count:6} ({pct:5.1}%)");
    }

    diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    println!("\nworst 8 elements (flat_idx, token, dim, burn, ort, |diff|):");
    for (i, b, r, d) in diffs.iter().take(8) {
        println!(
            "  {:6} tok {:3} dim {:3}: burn {:+.6} ort {:+.6} |d| {:.6}",
            i,
            i / 384,
            i % 384,
            b,
            r,
            d
        );
    }
    Ok(())
}
