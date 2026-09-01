//! Minimal demo: embed a few texts with the burn int8 E5 pipeline.
//!
//! Run with: cargo run --release -p e5-embed

use e5_embed::{E5Embedder, current_rss_mb, default_model_dir};

fn main() -> anyhow::Result<()> {
    let device = burn::prelude::Device::default();
    let model_dir = default_model_dir();

    println!("Loading int8 E5-small from {}...", model_dir.display());
    let embedder = E5Embedder::load(&model_dir, &device)?;
    println!("Model loaded. RSS: {:.1} MB", current_rss_mb());

    let passages = [
        "周末滨江夜骑 V11，速度很快！",
        "Hello world! Night ride along the river.",
        "夜骑",
    ];
    let vectors = embedder.embed_passages(&passages)?;

    for (text, v) in passages.iter().zip(&vectors) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "{text:?} -> dim={}, norm={norm:.6}, first4={:?}",
            v.len(),
            &v[..4]
        );
    }
    println!("RSS after inference: {:.1} MB", current_rss_mb());
    Ok(())
}
