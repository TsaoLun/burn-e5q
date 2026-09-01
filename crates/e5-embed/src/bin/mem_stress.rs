//! Memory stress: replay worst-case forwards under the 4096-token budget and
//! watch resident memory, mirroring inmotion-social's 512 MB container limit.
//!
//! Worst case per `pack_batches`: rows × padded_len ≤ 4096 with 512-token rows
//! → 8 rows × 512 tokens per forward. Repeat to see whether RSS ratchets.
//!
//! Run: `cargo run --release -p e5-embed --bin mem_stress`
//! or wrap with `/usr/bin/time -l` for the kernel-reported peak footprint.

use e5_embed::{E5Embedder, current_rss_mb, default_model_dir};

fn main() -> anyhow::Result<()> {
    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let budget: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    println!("RSS at start: {:.1} MB", current_rss_mb());
    let device = burn::prelude::Device::default();
    let embedder = E5Embedder::load(&default_model_dir(), &device)?
        .with_max_batch_tokens(budget);
    println!("RSS after model load: {:.1} MB", current_rss_mb());

    // Largest forward the budget allows: (budget/512) rows × ~512 tokens.
    let long_text: String = "Night ride along the riverside with friends. 周末滨江夜骑。 ".repeat(55);
    let rows = (budget / 512).max(1);
    let texts: Vec<&str> = std::iter::repeat_n(long_text.as_str(), rows).collect();
    let probe = embedder.encode_prefixed("passage: ", &long_text)?;
    println!(
        "stress batch: {rows} rows x {} tokens (budget {budget})",
        probe.len().min(512)
    );

    let mut peak = 0.0f64;
    for round in 0..rounds {
        let t = std::time::Instant::now();
        let out = embedder.embed_passages(&texts)?;
        let rss = current_rss_mb();
        peak = peak.max(rss);
        println!(
            "round {round:2}: {:8.1} ms, RSS {rss:7.1} MB ({} vectors)",
            t.elapsed().as_secs_f64() * 1e3,
            out.len()
        );
    }
    println!("\npeak observed RSS: {peak:.1} MB (container budget: 512 MB)");
    println!(
        "verdict: {}",
        if peak <= 512.0 {
            "within budget"
        } else {
            "EXCEEDS budget — see notes/poc-results.md"
        }
    );
    Ok(())
}
