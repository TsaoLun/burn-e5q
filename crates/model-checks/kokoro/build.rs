use burn_onnx::ModelGen;

fn main() {
    let artifacts = model_checks_common::artifacts_dir_build("kokoro");

    // KOKORO_DEBUG=1 generates the debug variant (kokoro-v1-debug.onnx) that
    // exposes intermediate tensors as graph outputs for bisection.
    let use_debug = std::env::var("KOKORO_DEBUG").ok().as_deref() == Some("1");
    let onnx_filename = if use_debug {
        "kokoro-v1-debug.onnx"
    } else {
        "kokoro-v1.0.onnx"
    };
    let onnx_path = artifacts.join(onnx_filename);

    println!("cargo:rerun-if-changed={}", onnx_path.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BURN_CACHE_DIR");
    println!("cargo:rerun-if-env-changed=KOKORO_DEBUG");

    if !onnx_path.exists() {
        eprintln!(
            "Error: ONNX model file not found at '{}'",
            onnx_path.display()
        );
        eprintln!();
        if use_debug {
            eprintln!("Run: uv run bisect_setup.py");
        } else {
            eprintln!("Run: uv run get_model.py");
        }
        std::process::exit(1);
    }

    ModelGen::new()
        .input(
            onnx_path
                .to_str()
                .expect("ONNX model path must be valid UTF-8"),
        )
        .out_dir("model/")
        .run_from_script();
}
