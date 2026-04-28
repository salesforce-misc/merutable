//! Integration test: compile examples/smoke.c and run it.
//!
//! This requires a C compiler (`cc`) on PATH.  The test is skipped when
//! `cc` is absent so the workspace builds cleanly on Rust-only CI images.
//!
//! Run explicitly:
//!   cargo nextest run -p merutable-capi --test c_smoke
//! or with the standard test runner:
//!   cargo test -p merutable-capi --test c_smoke

use std::{
    env,
    path::PathBuf,
    process::Command,
};

/// Locate the Cargo-built dylib/staticlib for merutable-capi.
///
/// Cargo sets CARGO_MANIFEST_DIR when running tests, and the built
/// artifacts land in <workspace>/target/<profile>/.
fn lib_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is the crate root at test time.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Walk up to workspace root.
    let workspace = manifest.parent().unwrap().parent().unwrap();
    // Test harness always runs in the "test" profile which maps to "debug".
    workspace.join("target").join("debug")
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Returns `false` when `cc` is not on PATH — lets CI skip gracefully.
fn cc_available() -> bool {
    Command::new("cc").arg("--version").output().is_ok()
}

#[test]
fn smoke_c_roundtrip() {
    if !cc_available() {
        eprintln!("cc not found — skipping C smoke test");
        return;
    }

    let lib = lib_dir();
    let root = crate_root();
    let include = root.join("include");
    let smoke_src = root.join("examples").join("smoke.c");
    let out_bin = lib.join("meru_smoke_test_bin");

    // ── Compile ──
    let status = Command::new("cc")
        .arg(&smoke_src)
        .arg(format!("-I{}", include.display()))
        .arg(format!("-L{}", lib.display()))
        .arg("-lmerutable_capi")
        .arg(format!("-Wl,-rpath,{}", lib.display()))
        .arg("-o")
        .arg(&out_bin)
        .status()
        .expect("cc invocation failed");
    assert!(status.success(), "C smoke test failed to compile");

    // ── Run ──
    // Use a temp dir so parallel test runs don't collide.
    let db_dir = env::temp_dir().join("meru_capi_smoke");
    let _ = std::fs::remove_dir_all(&db_dir);   // best-effort cleanup from prior run

    let out = Command::new(&out_bin)
        .env("MERU_SMOKE_DB", &db_dir)
        .output()
        .expect("smoke binary failed to launch");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    if !out.status.success() {
        eprintln!("stdout:\n{stdout}");
        eprintln!("stderr:\n{stderr}");
        panic!("C smoke test exited with status {}", out.status);
    }

    assert!(
        stdout.contains("smoke test PASSED"),
        "expected 'smoke test PASSED' in output, got:\n{stdout}"
    );

    // Best-effort cleanup.
    let _ = std::fs::remove_dir_all(&db_dir);
}
