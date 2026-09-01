use std::path::PathBuf;

use burn_onnx::ModelGen;

fn main() {
    println!("cargo:rerun-if-env-changed=E5_MODEL_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    let onnx_path = resolve_onnx_path();
    println!("cargo:rerun-if-changed={}", onnx_path.display());

    if !onnx_path.is_file() {
        eprintln!("ONNX model not found: {}", onnx_path.display());
        eprintln!(
            "Set E5_MODEL_PATH to model_qint8_avx512_vnni.onnx, or place it at crates/e5-embed/models/, or clone inmotion-social as a sibling of burn-e5q."
        );
        std::process::exit(1);
    }

    ModelGen::new()
        .input(onnx_path.to_str().expect("utf-8 onnx path"))
        .out_dir("model/")
        .development(true)
        .run_from_script();
}

fn resolve_onnx_path() -> PathBuf {
    if let Ok(p) = std::env::var("E5_MODEL_PATH") {
        return PathBuf::from(p);
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let candidates = [
        manifest_dir.join("models/model_qint8_avx512_vnni.onnx"),
        manifest_dir.join("../../../inmotion-social/data/models/multilingual-e5-small/model_qint8_avx512_vnni.onnx"),
    ];
    for path in &candidates {
        if path.is_file() {
            return path.clone();
        }
    }
    candidates[0].clone()
}
