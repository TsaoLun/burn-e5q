//! Live ONNX Runtime baseline via the Rust `ort` crate (no Burn, no Python).
//!
//! Matches inmotion-social: CPU EP, arena off by default, 4 intra-op threads.
//! Token ids come from `e5-embed/ref_data.json` (same cases as `compare_ort`).
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
use serde::Deserialize;

const TEXT_EMB_DIM: usize = 384;
const PAD_ID: i64 = 1;
const DEFAULT_PACK_BUDGET: usize = 4096;

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
        Some(hwm) => println!("{label:36}  RSS {rss:7.1} MB   HWM {hwm:7.1} MB"),
        None => println!("{label:36}  RSS {rss:7.1} MB"),
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

fn ref_path() -> PathBuf {
    if let Ok(p) = std::env::var("E5_REF_PATH") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../e5-embed/ref_data.json")
}

#[derive(Debug, Deserialize)]
struct RefCase {
    prefix: String,
    text: String,
    ids: Vec<i64>,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct RefData {
    model: String,
    cases: Vec<RefCase>,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn mean_pool_l2(
    hidden: &[f32],
    attn: &[i64],
    rows: usize,
    seq: usize,
    dim: usize,
) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0f32; dim]; rows];
    for r in 0..rows {
        let mut count = 0.0f32;
        for t in 0..seq {
            if attn[r * seq + t] == 0 {
                continue;
            }
            count += 1.0;
            let src = (r * seq + t) * dim;
            for d in 0..dim {
                out[r][d] += hidden[src + d];
            }
        }
        let denom = if count == 0.0 { 1.0 } else { count };
        let mut norm2 = 0.0f32;
        for d in 0..dim {
            out[r][d] /= denom;
            norm2 += out[r][d] * out[r][d];
        }
        let n = norm2.sqrt().max(1e-12);
        for d in 0..dim {
            out[r][d] /= n;
        }
    }
    out
}

fn pack_batches(row_lens: &[usize], max_batch_tokens: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..row_lens.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(row_lens[i]));

    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut padded_len = 0usize;
    for idx in order {
        let fits = !current.is_empty()
            && current.len() < 256
            && (current.len() + 1) * padded_len <= max_batch_tokens;
        if current.is_empty() {
            padded_len = row_lens[idx].max(1);
            current.push(idx);
        } else if fits {
            current.push(idx);
        } else {
            batches.push(std::mem::take(&mut current));
            padded_len = row_lens[idx].max(1);
            current.push(idx);
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

struct Batch {
    ids: Vec<i64>,
    attn: Vec<i64>,
    rows: usize,
    seq: usize,
}

fn pad_rows(rows: &[&[i64]]) -> Batch {
    let n = rows.len();
    let seq = rows.iter().map(|r| r.len()).max().unwrap_or(0).max(1);
    let mut ids = vec![PAD_ID; n * seq];
    let mut attn = vec![0i64; n * seq];
    for (i, row) in rows.iter().enumerate() {
        let start = i * seq;
        ids[start..start + row.len()].copy_from_slice(row);
        attn[start..start + row.len()].fill(1);
    }
    Batch {
        ids,
        attn,
        rows: n,
        seq,
    }
}

fn dummy_batch(rows: usize, seq: usize) -> Batch {
    let mut ids = vec![1i64; rows * seq];
    let attn = vec![1i64; rows * seq];
    for r in 0..rows {
        ids[r * seq] = 0;
        ids[r * seq + seq - 1] = 2;
    }
    Batch {
        ids,
        attn,
        rows,
        seq,
    }
}

fn run_hidden(session: &mut Session, batch: &Batch) -> anyhow::Result<(Vec<i64>, Vec<f32>)> {
    let n = batch.rows * batch.seq;
    let ids = Tensor::from_array(([batch.rows, batch.seq], batch.ids.clone())).context("ids")?;
    let attn = Tensor::from_array(([batch.rows, batch.seq], batch.attn.clone())).context("attn")?;
    let ttype = Tensor::from_array(([batch.rows, batch.seq], vec![0i64; n])).context("ttype")?;
    let outputs = session
        .run(ort::inputs![
            "input_ids" => ids,
            "attention_mask" => attn,
            "token_type_ids" => ttype,
        ])
        .context("session.run")?;
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let dims = shape.to_vec();
    if dims.len() != 3 {
        bail!("last_hidden rank {} want 3: {dims:?}", dims.len());
    }
    Ok((dims, data.to_vec()))
}

fn embed_batch(session: &mut Session, batch: &Batch) -> anyhow::Result<Vec<Vec<f32>>> {
    let (dims, hidden) = run_hidden(session, batch)?;
    let rows = dims[0] as usize;
    let seq = dims[1] as usize;
    let dim = dims[2] as usize;
    if dim != TEXT_EMB_DIM {
        bail!("hidden dim {dim} want {TEXT_EMB_DIM}");
    }
    Ok(mean_pool_l2(&hidden, &batch.attn, rows, seq, dim))
}

fn embed_packed(
    session: &mut Session,
    rows: &[&[i64]],
    budget: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let lens: Vec<usize> = rows.iter().map(|r| r.len()).collect();
    let mut out = vec![Vec::new(); rows.len()];
    for batch_idx in pack_batches(&lens, budget) {
        let slice: Vec<&[i64]> = batch_idx.iter().map(|&i| rows[i]).collect();
        let batch = pad_rows(&slice);
        let vecs = embed_batch(session, &batch)?;
        for (&i, v) in batch_idx.iter().zip(vecs) {
            out[i] = v;
        }
    }
    Ok(out)
}

fn bench_ms(
    session: &mut Session,
    repeat: usize,
    mut f: impl FnMut(&mut Session) -> anyhow::Result<()>,
) -> anyhow::Result<f64> {
    let mut best = f64::INFINITY;
    for _ in 0..repeat {
        let t0 = Instant::now();
        f(session)?;
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(best)
}

fn print_perf(label: &str, ms: f64, n_q: usize, n_tok: usize) {
    let qps = n_q as f64 * 1e3 / ms.max(1e-9);
    let tps = n_tok as f64 * 1e3 / ms.max(1e-9);
    println!("  {label}: {ms:8.1} ms   {qps:7.2} q/s   {tps:8.0} tok/s   ({n_tok} tok / {n_q} q)");
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
    let refs = ref_path();
    if !model.is_file() {
        bail!("missing ONNX at {}", model.display());
    }
    if !refs.is_file() {
        bail!("missing ref_data at {}", refs.display());
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
        "session loaded in {:.0} ms  arena={}  onnx file {:.1} MB",
        t0.elapsed().as_secs_f64() * 1e3,
        if arena { "on" } else { "off" },
        file_mb(&model)
    );
    log("after commit_from_file");

    let ref_data: RefData = serde_json::from_slice(&std::fs::read(&refs)?)?;
    println!("reference: {}  ({})", ref_data.model, refs.display());

    println!("\n=== Embedding cosine (Rust ort vs stored ref_data) ===");
    let mut min_cos = f32::INFINITY;
    let mut sum_cos = 0.0f64;
    let mut n_cos = 0usize;
    for case in &ref_data.cases {
        if case.text.trim().is_empty() {
            continue;
        }
        let batch = pad_rows(&[&case.ids]);
        let emb = embed_batch(&mut session, &batch)?;
        let cos = cosine(&emb[0], &case.embedding);
        min_cos = min_cos.min(cos);
        sum_cos += cos as f64;
        n_cos += 1;
        let mark = if cos > 0.999 { "✓" } else { "✗" };
        let label: String = case.text.chars().take(28).collect();
        println!(
            "  {mark} cos={cos:.6}  {}{:?}",
            case.prefix.trim_end_matches(": "),
            label
        );
    }
    println!(
        "  min cos = {min_cos:.6}, mean cos = {:.6} over {n_cos} cases",
        sum_cos / n_cos.max(1) as f64
    );
    log("after cosine cases");

    let passages: Vec<&RefCase> = ref_data
        .cases
        .iter()
        .filter(|c| c.prefix == "passage: " && !c.text.trim().is_empty())
        .collect();
    let short = passages
        .first()
        .context("no non-empty passage in ref_data")?;
    let long = passages
        .iter()
        .max_by_key(|c| c.ids.len())
        .copied()
        .context("no long passage")?;
    let passage_rows: Vec<&[i64]> = passages.iter().map(|c| c.ids.as_slice()).collect();
    let batch_toks: usize = passage_rows.iter().map(|r| r.len()).sum();

    // Warmup so the first timed run is not the session JIT/scratch allocate.
    let _ = embed_batch(&mut session, &pad_rows(&[&short.ids]))?;

    println!(
        "\n=== Latency (Rust ort, arena={}) ===",
        if arena { "on" } else { "off" }
    );
    let short_ms = bench_ms(&mut session, 5, |s| {
        embed_batch(s, &pad_rows(&[&short.ids]))?;
        Ok(())
    })?;
    print_perf("single short passage ", short_ms, 1, short.ids.len());

    let packed_ms = bench_ms(&mut session, 3, |s| {
        embed_packed(s, &passage_rows, DEFAULT_PACK_BUDGET)?;
        Ok(())
    })?;
    print_perf(
        "batch packed (burn-like)",
        packed_ms,
        passages.len(),
        batch_toks,
    );

    let padded = pad_rows(&passage_rows);
    let padded_ms = bench_ms(&mut session, 3, |s| {
        embed_batch(s, &padded)?;
        Ok(())
    })?;
    print_perf(
        "batch padded (one fwd) ",
        padded_ms,
        passages.len(),
        padded.rows * padded.seq,
    );

    let long_ms = bench_ms(&mut session, 3, |s| {
        embed_batch(s, &pad_rows(&[&long.ids]))?;
        Ok(())
    })?;
    print_perf("single long (512 tok)", long_ms, 1, long.ids.len());
    log("after compare_ort-equivalent");

    let rows = (budget / 512).max(1);
    println!("\n=== mem_stress {rows}×512, {rounds} rounds (budget {budget}) ===");
    let stress = dummy_batch(rows, 512);
    let mut peak = 0.0f64;
    let mut peak_hwm = 0.0f64;
    for round in 0..rounds {
        let t0 = Instant::now();
        let _ = run_hidden(&mut session, &stress)?;
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
    log("end");
    Ok(())
}
