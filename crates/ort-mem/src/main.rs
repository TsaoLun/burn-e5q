//! Entire ONNX Runtime process RSS (Rust `ort` crate, no Burn).
//!
//! Same dummy 16-tok / 512-tok / `budget÷512 × 512` shapes as
//! `e5-embed mem_stress` and `scripts/ort_mem.py`.
//!
//! ```text
//! cargo run --release -p ort-mem -- -- 5 2048
//! cargo run --release -p ort-mem -- --arena -- 5 2048
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, bail};
use ort::ep::CPU;
use ort::session::Session;
use ort::value::Tensor;

/// `ort::Error<SessionBuilder>` is not `Send`, so it cannot go through `anyhow::Error` via `?`.
fn o<T>(r: Result<T, impl std::fmt::Display>) -> anyhow::Result<T> {
    r.map_err(|e| anyhow::anyhow!("{e}"))
}

fn rss_hwm() -> (f64, Option<f64>) {
    let text = std::fs::read_to_string("/proc/self/status").ok();
    let mut rss = 0.0;
    let mut hwm = None;
    if let Some(text) = text {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Some(kb) = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<f64>().ok())
                {
                    rss = kb / 1024.0;
                }
            } else if let Some(rest) = line.strip_prefix("VmHWM:") {
                hwm = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(|kb| kb / 1024.0);
            }
        }
    }
    (rss, hwm)
}

fn log(label: &str) {
    let (rss, hwm) = rss_hwm();
    match hwm {
        Some(hwm) => println!("{label:28}  RSS {rss:7.1} MB   HWM {hwm:7.1} MB"),
        None => println!("{label:28}  RSS {rss:7.1} MB"),
    }
}

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("E5_MODEL_PATH") {
        return PathBuf::from(p);
    }
    if let Ok(dir) = std::env::var("E5_MODEL_DIR") {
        return PathBuf::from(dir).join("model_qint8_avx512_vnni.onnx");
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../e5-embed/models/model_qint8_avx512_vnni.onnx")
}

fn make_ids(rows: usize, seq: usize) -> Vec<i64> {
    let mut ids = vec![1i64; rows * seq];
    for r in 0..rows {
        ids[r * seq] = 0;
        ids[r * seq + seq - 1] = 2;
    }
    ids
}

fn run_forward(session: &mut Session, rows: usize, seq: usize) -> anyhow::Result<()> {
    let n = rows * seq;
    let ids = Tensor::from_array(([rows, seq], make_ids(rows, seq))).context("input_ids tensor")?;
    let attn = Tensor::from_array(([rows, seq], vec![1i64; n])).context("attention_mask tensor")?;
    let ttype =
        Tensor::from_array(([rows, seq], vec![0i64; n])).context("token_type_ids tensor")?;
    let _outputs = session
        .run(ort::inputs![
            "input_ids" => ids,
            "attention_mask" => attn,
            "token_type_ids" => ttype,
        ])
        .context("session.run")?;
    Ok(())
}

fn parse_args() -> anyhow::Result<(usize, usize, bool)> {
    let mut rounds = 5usize;
    let mut budget = 2048usize;
    let mut arena = false;
    let mut rest: Vec<String> = Vec::new();
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "--arena" => arena = true,
            "--" => {}
            _ if a.starts_with('-') => bail!("unknown flag {a}"),
            _ => rest.push(a),
        }
    }
    if let Some(s) = rest.first() {
        rounds = s.parse().context("rounds")?;
    }
    if let Some(s) = rest.get(1) {
        budget = s.parse().context("budget")?;
    }
    Ok((rounds, budget, arena))
}

fn file_mb(path: &Path) -> f64 {
    path.metadata()
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}

fn main() -> anyhow::Result<()> {
    let (rounds, budget, arena) = parse_args()?;
    let model = model_path();
    if !model.is_file() {
        bail!("missing ONNX at {}", model.display());
    }

    log("native start");

    let builder = o(Session::builder())?;
    let builder = o(builder.with_intra_threads(4))?;
    let builder = o(builder.with_inter_threads(1))?;
    let builder = o(builder.with_memory_pattern(arena))?;
    let mut builder =
        o(builder.with_execution_providers([CPU::default().with_arena_allocator(arena).build()]))?;

    log("after Session::builder");
    let t0 = Instant::now();
    let mut session = o(builder.commit_from_file(&model))?;
    println!(
        "session loaded in {:.0} ms  arena={}  onnx file {:.1} MB  {}",
        t0.elapsed().as_secs_f64() * 1e3,
        if arena { "on" } else { "off" },
        file_mb(&model),
        model.display()
    );
    log("after commit_from_file");

    let rows = (budget / 512).max(1);
    println!("stress batch: {rows} rows x 512 tokens (budget {budget})");

    let t0 = Instant::now();
    run_forward(&mut session, 1, 16)?;
    println!("single 16 tok: {:8.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    log("after 16-tok forward");

    let t0 = Instant::now();
    run_forward(&mut session, 1, 512)?;
    println!(
        "single 512 tok: {:8.1} ms",
        t0.elapsed().as_secs_f64() * 1e3
    );
    log("after 512-tok forward");

    let mut peak = 0.0f64;
    let mut peak_hwm = 0.0f64;
    for round in 0..rounds {
        let t0 = Instant::now();
        run_forward(&mut session, rows, 512)?;
        let (rss, hwm) = rss_hwm();
        peak = peak.max(rss);
        if let Some(h) = hwm {
            peak_hwm = peak_hwm.max(h);
        }
        match hwm {
            Some(hwm) => println!(
                "round {round:2}: {:8.1} ms, RSS {rss:7.1} MB  HWM {hwm:7.1} MB",
                t0.elapsed().as_secs_f64() * 1e3
            ),
            None => println!(
                "round {round:2}: {:8.1} ms, RSS {rss:7.1} MB",
                t0.elapsed().as_secs_f64() * 1e3
            ),
        }
    }
    println!("\npeak observed RSS: {peak:.1} MB (container budget: 512 MB)");
    if peak_hwm > 0.0 {
        println!("kernel peak HWM:    {peak_hwm:.1} MB");
    }
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
