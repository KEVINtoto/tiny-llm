//! Shared temporary directories rooted in this checkout, independent of cwd/TMPDIR.
#![allow(dead_code)]

use std::path::PathBuf;

pub fn root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tmp");
    std::fs::create_dir_all(&root).expect("create project temporary directory");
    root
}

pub fn tempdir() -> std::io::Result<tempfile::TempDir> {
    tempfile::Builder::new()
        .prefix("tiny-llm-agent-")
        .tempdir_in(root())
}
