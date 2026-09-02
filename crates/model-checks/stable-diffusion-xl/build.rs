use burn_onnx::ModelGen;

fn main() {
    let artifacts = model_checks_common::artifacts_dir_build("stable-diffusion-xl");
    let onnx_path = artifacts.join("sdxl-unet.onnx");
    let test_data_path = artifacts.join("test_data.pt");

    // SDXL UNet uses ONNX external data format: model.onnx + model.onnx_data
    let onnx_data_path = artifacts.join("model.onnx_data");

    // Tell Cargo to only rebuild if these files change
    println!("cargo:rerun-if-changed={}", onnx_path.display());
    println!("cargo:rerun-if-changed={}", onnx_data_path.display());
    println!("cargo:rerun-if-changed={}", test_data_path.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BURN_CACHE_DIR");

    // Check if the ONNX model and external data files exist
    if !onnx_path.exists() || !onnx_data_path.exists() {
        eprintln!(
            "Error: ONNX model files not found at '{}'",
            artifacts.display()
        );
        eprintln!();
        eprintln!("Please run the following command to download and prepare the model:");
        eprintln!("  uv run get_model.py");
        eprintln!();
        eprintln!("This will download the SDXL UNet ONNX model and generate test data.");
        std::process::exit(1);
    }

    // Generate the model code from the ONNX file
    ModelGen::new()
        .input(
            onnx_path
                .to_str()
                .expect("ONNX model path must be valid UTF-8"),
        )
        .out_dir("model/")
        .run_from_script();
}
