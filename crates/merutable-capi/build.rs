fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_path = std::path::PathBuf::from(&crate_dir)
        .join("include")
        .join("merutable.h");

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(
            cbindgen::Config::from_file(
                std::path::PathBuf::from(&crate_dir).join("cbindgen.toml"),
            )
            .expect("cbindgen.toml"),
        )
        .generate()
        .expect("cbindgen failed")
        .write_to_file(&out_path);

    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
