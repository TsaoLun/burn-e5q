use burn_onnx::ModelGen;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let onnx_path = format!("{manifest_dir}/models/dql_matmul.onnx");

    println!("cargo:rerun-if-changed={onnx_path}");
    println!("cargo:rerun-if-changed=build.rs");

    if !std::path::Path::new(&onnx_path).exists() {
        eprintln!("ONNX model not found: {onnx_path}");
        eprintln!("Run: python3 crates/dql-poc/models/make_dql_model.py");
        std::process::exit(1);
    }

    ModelGen::new()
        .input(&onnx_path)
        .out_dir("model/")
        .development(true)
        .run_from_script();
}
