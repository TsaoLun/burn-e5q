//! Isolate remaining 512-token latency after fused attention + head-parallel flash.
//!
//! Same flex backend and e5 shapes as `compare_ort`. Times each leftover block
//! alone, then a chained 12-layer skeleton, then one real 512-token forward.
//! Does not change the model.
//!
//! ```bash
//! cargo run --release -p e5-embed --bin breakdown
//! ```

use std::hint::black_box;
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, Distribution, TensorData};

use e5_embed::{E5Embedder, default_model_dir};

const HEADS: usize = 12;
const HEAD_DIM: usize = 32;
const HIDDEN: usize = 384;
const FFN: usize = 1536;
const LAYERS: usize = 12;
const VOCAB: usize = 250_037;
const ATTN_SCALE: f64 = 0.1767766922712326; // 1/√32

fn main() -> anyhow::Result<()> {
    let device = Device::default();
    println!(
        "Device: {device:?} ({})",
        if cfg!(feature = "cpu") {
            "cubecl-cpu"
        } else {
            "flex"
        }
    );
    print_cpu_line();

    println!("\n=== Isolated kernels (min of repeats; flex is eager) ===");
    println!(
        "  {:<28} {:>10} {:>10} {:>10} {:>8}",
        "block", "min ms", "med ms", "per-call", "GOPS"
    );

    let s512 = Shapes::new(512);
    let s16 = Shapes::new(16);

    let r512 = run_shape(&device, &s512, 5);

    println!("\n=== Same kernels at 16 tok (naive attention; dispatch-heavy) ===");
    let r16 = run_shape(&device, &s16, 8);

    println!("\n=== Chained 12-layer skeleton vs isolated sum ===");
    println!(
        "  512 isolated sum (MMI+attn+DQL+LN+GELU+dequant+glue+embed): {:7.1} ms",
        r512.accounted()
    );
    println!(
        "  512 assembled 12 layers (dummy weights, same op order):     {:7.1} ms",
        r512.assembled_ms
    );
    println!(
        "  16  isolated sum:                                           {:7.1} ms",
        r16.accounted()
    );
    println!(
        "  16  assembled 12 layers:                                    {:7.1} ms",
        r16.assembled_ms
    );

    println!("\n=== Calibration: real E5Embedder forward ===");
    let embedder = E5Embedder::load(&default_model_dir(), &device)?;

    let long: String = "Night ride along the riverside with friends. 周末滨江夜骑。 ".repeat(55);
    let long_ids = embedder.encode_prefixed("passage: ", &long)?;
    let long_toks = long_ids.len().min(512);
    let short = "周末滨江夜骑 V11，速度很快！";
    let short_ids = embedder.encode_prefixed("passage: ", short)?;
    let short_toks = short_ids.len().min(512);

    let full_512 = time_min_ms(3, || {
        let _ = black_box(embedder.embed_passages(&[&long]).expect("512 embed"));
    });
    let full_16 = time_min_ms(5, || {
        let _ = black_box(embedder.embed_passages(&[short]).expect("16 embed"));
    });

    println!(
        "  real 512-tok embed_passages ({long_toks} tok, min of 3): {full_512:7.1} ms"
    );
    println!(
        "  real 16-tok  embed_passages ({short_toks} tok, min of 5): {full_16:7.1} ms"
    );
    println!(
        "  512 unaccounted (real − isolated sum):  {:>7.1} ms ({:.0}%)",
        full_512 - r512.accounted(),
        100.0 * (full_512 - r512.accounted()) / full_512.max(1e-9)
    );
    println!(
        "  512 unaccounted (real − assembled):     {:>7.1} ms ({:.0}%)",
        full_512 - r512.assembled_ms,
        100.0 * (full_512 - r512.assembled_ms) / full_512.max(1e-9)
    );
    println!(
        "  16  unaccounted (real − isolated sum):  {:>7.1} ms ({:.0}%)",
        full_16 - r16.accounted(),
        100.0 * (full_16 - r16.accounted()) / full_16.max(1e-9)
    );

    println!("\n=== Share of real 512 forward (isolated min / real) ===");
    print_share("flash attention ×12", r512.flash_ms, full_512);
    print_share("MMI QKV+out ×36", r512.mmi_qkv_ms, full_512);
    print_share("MMI FFN1 ×12", r512.mmi_ffn1_ms, full_512);
    print_share("MMI FFN2 ×12", r512.mmi_ffn2_ms, full_512);
    print_share("MMI all 72", r512.mmi_all(), full_512);
    print_share("DQL ×48", r512.dql_ms, full_512);
    print_share("expanded LN ×25", r512.ln_ms, full_512);
    print_share("expanded GELU ×12", r512.gelu_ms, full_512);
    print_share("MMI dequant ×72", r512.dequant_ms, full_512);
    print_share("QKV reshape/permute ×12", r512.glue_ms, full_512);
    print_share("embedding take+dequant", r512.embed_ms, full_512);
    print_share("isolated sum", r512.accounted(), full_512);
    print_share("assembled 12 layers", r512.assembled_ms, full_512);
    print_share("Rust ort 512 (54 ms)", 53.8, full_512);

    println!("\nHow to read:");
    println!("  MMI dominates            → next cut is FFN/QKV schedule or fused layer, not flash");
    println!("  LN+GELU+glue large       → codegen layer_norm/gelu or a whole-layer exec unit");
    println!("  isolated ≪ real          → leftover is generated-graph tax (val/shape/clone)");
    println!("  isolated ≈ real, MMI big → ORT wins on fusion, not a single slow kernel");
    Ok(())
}

struct Shapes {
    seq: usize,
    qkv: [usize; 4],
    hidden: [usize; 3],
    ffn: [usize; 3],
    bias: [usize; 4],
}

impl Shapes {
    fn new(seq: usize) -> Self {
        Self {
            seq,
            qkv: [1, HEADS, seq, HEAD_DIM],
            hidden: [1, seq, HIDDEN],
            ffn: [1, seq, FFN],
            bias: [1, 1, 1, seq],
        }
    }
}

struct ShapeResult {
    flash_ms: f64,
    mmi_qkv_ms: f64,
    mmi_ffn1_ms: f64,
    mmi_ffn2_ms: f64,
    dql_ms: f64,
    ln_ms: f64,
    gelu_ms: f64,
    dequant_ms: f64,
    glue_ms: f64,
    embed_ms: f64,
    assembled_ms: f64,
}

impl ShapeResult {
    fn mmi_all(&self) -> f64 {
        self.mmi_qkv_ms + self.mmi_ffn1_ms + self.mmi_ffn2_ms
    }

    fn accounted(&self) -> f64 {
        self.flash_ms
            + self.mmi_all()
            + self.dql_ms
            + self.ln_ms
            + self.gelu_ms
            + self.dequant_ms
            + self.glue_ms
            + self.embed_ms
    }
}

fn run_shape(device: &Device, shapes: &Shapes, repeats: usize) -> ShapeResult {
    let seq = shapes.seq;
    let mac_qkv = (seq * HIDDEN * HIDDEN) as f64;
    let mac_ffn1 = (seq * HIDDEN * FFN) as f64;
    let mac_ffn2 = (seq * FFN * HIDDEN) as f64;
    // flash: 12 heads × (QK + PV) ≈ 2 × H × S × S × D
    let mac_flash = (2 * HEADS * seq * seq * HEAD_DIM) as f64;

    let flash_ms = time_flash(device, shapes, repeats);
    print_row(
        &format!("flash attn ×{LAYERS} ({seq})"),
        flash_ms,
        LAYERS,
        Some(mac_flash * LAYERS as f64),
    );

    let mmi_qkv_ms = time_mmi_block(device, shapes.hidden, [1, HIDDEN, HIDDEN], 36, repeats);
    print_row(
        &format!("MMI QKV+out ×36 ({seq}×384×384)"),
        mmi_qkv_ms,
        36,
        Some(mac_qkv * 36.0),
    );

    let mmi_ffn1_ms = time_mmi_block(device, shapes.hidden, [1, HIDDEN, FFN], 12, repeats);
    print_row(
        &format!("MMI FFN1 ×12 ({seq}×384×1536)"),
        mmi_ffn1_ms,
        12,
        Some(mac_ffn1 * 12.0),
    );

    let mmi_ffn2_ms = time_mmi_block(device, shapes.ffn, [1, FFN, HIDDEN], 12, repeats);
    print_row(
        &format!("MMI FFN2 ×12 ({seq}×1536×384)"),
        mmi_ffn2_ms,
        12,
        Some(mac_ffn2 * 12.0),
    );

    let dql_ms = time_dql(device, shapes, repeats);
    print_row(&format!("DQL ×48 ({seq})"), dql_ms, 48, None);

    let ln_ms = time_ln(device, shapes, repeats);
    print_row(&format!("expanded LN ×25 ({seq})"), ln_ms, 25, None);

    let gelu_ms = time_gelu(device, shapes, repeats);
    print_row(&format!("expanded GELU ×12 ({seq})"), gelu_ms, 12, None);

    let dequant_ms = time_dequant(device, shapes, repeats);
    print_row(&format!("MMI dequant ×72 ({seq})"), dequant_ms, 72, None);

    let glue_ms = time_glue(device, shapes, repeats);
    print_row(
        &format!("QKV reshape/perm ×12 ({seq})"),
        glue_ms,
        12,
        None,
    );

    let embed_ms = time_embed(device, seq, repeats);
    print_row(&format!("embed take+dequant ({seq})"), embed_ms, 1, None);

    let assembled_ms = time_assembled(device, shapes, repeats);
    print_row(
        &format!("assembled 12 layers ({seq})"),
        assembled_ms,
        LAYERS,
        None,
    );

    ShapeResult {
        flash_ms: flash_ms.0,
        mmi_qkv_ms: mmi_qkv_ms.0,
        mmi_ffn1_ms: mmi_ffn1_ms.0,
        mmi_ffn2_ms: mmi_ffn2_ms.0,
        dql_ms: dql_ms.0,
        ln_ms: ln_ms.0,
        gelu_ms: gelu_ms.0,
        dequant_ms: dequant_ms.0,
        glue_ms: glue_ms.0,
        embed_ms: embed_ms.0,
        assembled_ms: assembled_ms.0,
    }
}

fn print_row(label: &str, (min, med): (f64, f64), calls: usize, macs: Option<f64>) {
    let per = min / calls.max(1) as f64;
    let gops = macs
        .map(|m| format!("{:.1}", m / (min.max(1e-9) * 1e6)))
        .unwrap_or_else(|| "—".into());
    println!("  {label:<28} {min:10.1} {med:10.1} {per:9.2} ms {gops:>8}");
}

fn print_share(label: &str, ms: f64, total: f64) {
    println!(
        "  {label:<28} {ms:7.1} ms  {:>5.1}%",
        100.0 * ms / total.max(1e-9)
    );
}

fn print_cpu_line() {
    let ok = std::fs::read_to_string("/proc/cpuinfo").ok();
    let Some(text) = ok else {
        return;
    };
    let cores = text.matches("processor").count();
    let flags = text
        .lines()
        .find(|l| l.starts_with("flags"))
        .unwrap_or("");
    let vnni = flags.contains("avx512_vnni");
    let amx = flags.contains("amx_int8");
    println!("CPU: {cores} logical, avx512_vnni={vnni}, amx_int8={amx}");
}

fn time_min_ms(iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn time_pair(iters: usize, mut f: impl FnMut()) -> (f64, f64) {
    f();
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (samples[0], samples[samples.len() / 2])
}

fn f32_tensor<const D: usize>(shape: [usize; D], lo: f64, hi: f64, device: &Device) -> Tensor<D> {
    Tensor::<D>::random(shape, Distribution::Uniform(lo, hi), device)
}

fn u8_act<const D: usize>(shape: [usize; D], device: &Device) -> Tensor<D, Int> {
    let n: usize = shape.iter().product();
    Tensor::<D, Int>::from_data(
        TensorData::new(vec![128u8; n], shape),
        (device, DType::U8),
    )
}

fn i8_w(rows: usize, cols: usize, device: &Device) -> Tensor<3, Int> {
    Tensor::<2, Int>::from_data(
        TensorData::new(vec![1i8; rows * cols], [rows, cols]),
        (device, DType::I8),
    )
    .unsqueeze::<3>()
}

fn zp_i32(device: &Device) -> Tensor<3, Int> {
    Tensor::<1, Int>::from_data(TensorData::new(vec![0i32], [1]), (device, DType::I32))
        .unsqueeze::<3>()
}

fn vec1(v: f32, device: &Device) -> Tensor<1> {
    Tensor::<1>::from_data(TensorData::new(vec![v], [1]), device)
}

fn vec_hidden(v: f32, device: &Device) -> Tensor<1> {
    Tensor::<1>::from_data(TensorData::new(vec![v; HIDDEN], [HIDDEN]), device)
}

fn attn_opts() -> AttentionModuleOptions {
    AttentionModuleOptions {
        scale: Some(ATTN_SCALE),
        softcap: None,
        is_causal: false,
    }
}

fn time_flash(device: &Device, shapes: &Shapes, repeats: usize) -> (f64, f64) {
    let q = f32_tensor(shapes.qkv, -0.5, 0.5, device);
    let k = f32_tensor(shapes.qkv, -0.5, 0.5, device);
    let v = f32_tensor(shapes.qkv, -0.5, 0.5, device);
    let bias = f32_tensor(shapes.bias, -2.0, 0.0, device);
    time_pair(repeats, || {
        for _ in 0..LAYERS {
            let y = attention(
                q.clone(),
                k.clone(),
                v.clone(),
                None,
                Some(bias.clone()),
                attn_opts(),
            );
            black_box(y);
        }
    })
}

fn time_mmi_block(
    device: &Device,
    act_shape: [usize; 3],
    w_shape: [usize; 3],
    calls: usize,
    repeats: usize,
) -> (f64, f64) {
    let act = u8_act(act_shape, device);
    let w = i8_w(w_shape[1], w_shape[2], device);
    let za = zp_i32(device);
    let zb = zp_i32(device);
    time_pair(repeats, || {
        for _ in 0..calls {
            let y = act.clone().matmul_integer(
                w.clone(),
                Some(za.clone()),
                Some(zb.clone()),
            );
            black_box(y);
        }
    })
}

fn time_dql(device: &Device, shapes: &Shapes, repeats: usize) -> (f64, f64) {
    let h = f32_tensor(shapes.hidden, -3.0, 3.0, device);
    let f = f32_tensor(shapes.ffn, -3.0, 3.0, device);
    time_pair(repeats, || {
        for _ in 0..36 {
            black_box(h.clone().dynamic_quantize_linear());
        }
        for _ in 0..12 {
            black_box(f.clone().dynamic_quantize_linear());
        }
    })
}

fn expanded_ln(x: Tensor<3>, gamma: &Tensor<1>, beta: &Tensor<1>, two: &Tensor<1>, eps: &Tensor<1>) -> Tensor<3> {
    let mean = x.clone().mean_dim(2);
    let centered = x.sub(mean);
    let var = centered
        .clone()
        .powf(two.clone().unsqueeze_dims(&[0isize, 1isize]))
        .mean_dim(2);
    let inv = var
        .add(eps.clone().unsqueeze_dims(&[0isize, 1isize]))
        .sqrt();
    let norm = centered.div(inv);
    norm.mul(gamma.clone().unsqueeze_dims(&[0isize, 1isize]))
        .add(beta.clone().unsqueeze_dims(&[0isize, 1isize]))
}

fn time_ln(device: &Device, shapes: &Shapes, repeats: usize) -> (f64, f64) {
    let x = f32_tensor(shapes.hidden, -2.0, 2.0, device);
    let gamma = vec_hidden(1.0, device);
    let beta = vec_hidden(0.0, device);
    let two = vec1(2.0, device);
    let eps = vec1(1e-5, device);
    time_pair(repeats, || {
        // 12 attn-pre + 12 ffn-pre + embedding/final ≈ 25
        for _ in 0..25 {
            black_box(expanded_ln(
                x.clone(),
                &gamma,
                &beta,
                &two,
                &eps,
            ));
        }
    })
}

fn expanded_gelu(x: Tensor<3>, inv_sqrt2: &Tensor<1>, one: &Tensor<1>, half: &Tensor<1>) -> Tensor<3> {
    let t = x
        .clone()
        .div(inv_sqrt2.clone().unsqueeze_dims(&[0isize, 1isize]));
    let e = t
        .erf()
        .add(one.clone().unsqueeze_dims(&[0isize, 1isize]));
    x.mul(e)
        .mul(half.clone().unsqueeze_dims(&[0isize, 1isize]))
}

fn time_gelu(device: &Device, shapes: &Shapes, repeats: usize) -> (f64, f64) {
    let x = f32_tensor(shapes.ffn, -2.0, 2.0, device);
    let inv_sqrt2 = vec1(std::f32::consts::FRAC_1_SQRT_2, device);
    let one = vec1(1.0, device);
    let half = vec1(0.5, device);
    time_pair(repeats, || {
        for _ in 0..12 {
            black_box(expanded_gelu(x.clone(), &inv_sqrt2, &one, &half));
        }
    })
}

fn time_dequant(device: &Device, shapes: &Shapes, repeats: usize) -> (f64, f64) {
    // i32 MMI output of QKV/out size, 60 of 72; FFN1/FFN2 covered separately below.
    let acc_h = Tensor::<3, Int>::from_data(
        TensorData::new(vec![0i32; shapes.seq * HIDDEN], [1, shapes.seq, HIDDEN]),
        (device, DType::I32),
    );
    let acc_f = Tensor::<3, Int>::from_data(
        TensorData::new(vec![0i32; shapes.seq * FFN], [1, shapes.seq, FFN]),
        (device, DType::I32),
    );
    let scale = vec1(0.02, device);
    let bias = vec_hidden(0.0, device);
    let bias_f = Tensor::<1>::from_data(TensorData::new(vec![0.0f32; FFN], [FFN]), device);
    time_pair(repeats, || {
        for _ in 0..48 {
            let y = acc_h
                .clone()
                .float()
                .cast(DType::F32)
                .mul(scale.clone().unsqueeze_dims(&[0isize, 1isize]))
                .add(bias.clone().unsqueeze_dims(&[0isize, 1isize]));
            black_box(y);
        }
        for _ in 0..12 {
            let y = acc_f
                .clone()
                .float()
                .cast(DType::F32)
                .mul(scale.clone().unsqueeze_dims(&[0isize, 1isize]))
                .add(bias_f.clone().unsqueeze_dims(&[0isize, 1isize]));
            black_box(y);
        }
        for _ in 0..12 {
            let y = acc_h
                .clone()
                .float()
                .cast(DType::F32)
                .mul(scale.clone().unsqueeze_dims(&[0isize, 1isize]))
                .add(bias.clone().unsqueeze_dims(&[0isize, 1isize]));
            black_box(y);
        }
    })
}

fn time_glue(device: &Device, shapes: &Shapes, repeats: usize) -> (f64, f64) {
    let qkv = f32_tensor(shapes.hidden, -1.0, 1.0, device);
    let attn_out = f32_tensor(shapes.qkv, -1.0, 1.0, device);
    time_pair(repeats, || {
        for _ in 0..LAYERS {
            let q = qkv.clone().reshape([1, shapes.seq, HEADS, HEAD_DIM]).permute([0, 2, 1, 3]);
            let k = qkv.clone().reshape([1, shapes.seq, HEADS, HEAD_DIM]).permute([0, 2, 3, 1]);
            let v = qkv.clone().reshape([1, shapes.seq, HEADS, HEAD_DIM]).permute([0, 2, 1, 3]);
            // K corrective transpose written into the fused-attn path
            let k = k.permute([0, 1, 3, 2]);
            let y = attn_out.clone().permute([0, 2, 1, 3]).reshape([1, shapes.seq, HIDDEN]);
            black_box((q, k, v, y));
        }
    })
}

fn time_embed(device: &Device, seq: usize, repeats: usize) -> (f64, f64) {
    let table = Tensor::<2, Int>::from_data(
        TensorData::new(vec![1u8; VOCAB * HIDDEN], [VOCAB, HIDDEN]),
        (device, DType::U8),
    );
    let ids: Vec<i32> = (0..seq)
        .map(|i| ((i as i32).wrapping_mul(7919)).rem_euclid(VOCAB as i32))
        .collect();
    let ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, seq]), device);
    let scale = vec1(0.03, device);
    let zp = Tensor::<1, Int>::from_data(TensorData::new(vec![128i32], [1]), (device, DType::I32));
    time_pair(repeats, || {
        let gathered = table.clone().take::<2, 3>(0, ids.clone());
        let y = gathered
            .cast(DType::I32)
            .float()
            .cast(DType::F32)
            .sub(zp.clone().float().cast(DType::F32).unsqueeze_dims(&[0isize, 2isize]))
            .mul(scale.clone().unsqueeze_dims(&[0isize, 2isize]));
        black_box(y);
    })
}

struct LayerWeights {
    w_q: Tensor<3, Int>,
    w_k: Tensor<3, Int>,
    w_v: Tensor<3, Int>,
    w_o: Tensor<3, Int>,
    w_f1: Tensor<3, Int>,
    w_f2: Tensor<3, Int>,
    zb: Tensor<3, Int>,
    gamma: Tensor<1>,
    beta: Tensor<1>,
    two: Tensor<1>,
    eps: Tensor<1>,
    inv_sqrt2: Tensor<1>,
    one: Tensor<1>,
    half: Tensor<1>,
    scale: Tensor<1>,
    bias_h: Tensor<1>,
    bias_f: Tensor<1>,
    attn_bias: Tensor<4>,
}

fn layer_weights(device: &Device, seq: usize) -> LayerWeights {
    LayerWeights {
        w_q: i8_w(HIDDEN, HIDDEN, device),
        w_k: i8_w(HIDDEN, HIDDEN, device),
        w_v: i8_w(HIDDEN, HIDDEN, device),
        w_o: i8_w(HIDDEN, HIDDEN, device),
        w_f1: i8_w(HIDDEN, FFN, device),
        w_f2: i8_w(FFN, HIDDEN, device),
        zb: zp_i32(device),
        gamma: vec_hidden(1.0, device),
        beta: vec_hidden(0.0, device),
        two: vec1(2.0, device),
        eps: vec1(1e-5, device),
        inv_sqrt2: vec1(std::f32::consts::FRAC_1_SQRT_2, device),
        one: vec1(1.0, device),
        half: vec1(0.5, device),
        scale: vec1(0.02, device),
        bias_h: vec_hidden(0.0, device),
        bias_f: Tensor::<1>::from_data(TensorData::new(vec![0.0f32; FFN], [FFN]), device),
        attn_bias: f32_tensor([1, 1, 1, seq], -2.0, 0.0, device),
    }
}

fn dequant_h(acc: Tensor<3, Int>, scale: &Tensor<1>, bias: &Tensor<1>) -> Tensor<3> {
    acc.float()
        .cast(DType::F32)
        .mul(scale.clone().unsqueeze_dims(&[0isize, 1isize]))
        .add(bias.clone().unsqueeze_dims(&[0isize, 1isize]))
}

fn one_layer(x: Tensor<3>, w: &LayerWeights, seq: usize) -> Tensor<3> {
    let x_ln = expanded_ln(x.clone(), &w.gamma, &w.beta, &w.two, &w.eps);
    let (u8_x, _s, zp) = x_ln.clone().dynamic_quantize_linear();
    let za = zp.cast(DType::I32).unsqueeze::<3>();
    let q = dequant_h(
        u8_x.clone()
            .matmul_integer(w.w_q.clone(), Some(za.clone()), Some(w.zb.clone())),
        &w.scale,
        &w.bias_h,
    );
    let k = dequant_h(
        u8_x.clone()
            .matmul_integer(w.w_k.clone(), Some(za.clone()), Some(w.zb.clone())),
        &w.scale,
        &w.bias_h,
    );
    let v = dequant_h(
        u8_x.matmul_integer(w.w_v.clone(), Some(za), Some(w.zb.clone())),
        &w.scale,
        &w.bias_h,
    );
    let q = q.reshape([1, seq, HEADS, HEAD_DIM]).permute([0, 2, 1, 3]);
    let k = k
        .reshape([1, seq, HEADS, HEAD_DIM])
        .permute([0, 2, 3, 1])
        .permute([0, 1, 3, 2]);
    let v = v.reshape([1, seq, HEADS, HEAD_DIM]).permute([0, 2, 1, 3]);
    let ctx = attention(q, k, v, None, Some(w.attn_bias.clone()), attn_opts());
    let ctx = ctx.permute([0, 2, 1, 3]).reshape([1, seq, HIDDEN]);
    let (u8_c, _s, zp) = ctx.dynamic_quantize_linear();
    let za = zp.cast(DType::I32).unsqueeze::<3>();
    let proj = dequant_h(
        u8_c.matmul_integer(w.w_o.clone(), Some(za), Some(w.zb.clone())),
        &w.scale,
        &w.bias_h,
    );
    let x = x.add(proj);

    let y_ln = expanded_ln(x.clone(), &w.gamma, &w.beta, &w.two, &w.eps);
    let (u8_y, _s, zp) = y_ln.dynamic_quantize_linear();
    let za = zp.cast(DType::I32).unsqueeze::<3>();
    let h = dequant_h(
        u8_y.matmul_integer(w.w_f1.clone(), Some(za), Some(w.zb.clone())),
        &w.scale,
        &w.bias_f,
    );
    let h = expanded_gelu(h, &w.inv_sqrt2, &w.one, &w.half);
    let (u8_h, _s, zp) = h.dynamic_quantize_linear();
    let za = zp.cast(DType::I32).unsqueeze::<3>();
    let out = dequant_h(
        u8_h.matmul_integer(w.w_f2.clone(), Some(za), Some(w.zb.clone())),
        &w.scale,
        &w.bias_h,
    );
    x.add(out)
}

fn time_assembled(device: &Device, shapes: &Shapes, repeats: usize) -> (f64, f64) {
    let w = layer_weights(device, shapes.seq);
    let x0 = f32_tensor(shapes.hidden, -1.0, 1.0, device);
    time_pair(repeats, || {
        let mut x = x0.clone();
        for _ in 0..LAYERS {
            x = one_layer(x, &w, shapes.seq);
        }
        black_box(x);
    })
}
