use std::{fs, io::Write, path::Path};

use sha2::{Digest, digest::Update};

use crate::AgentError;

pub fn get_sha256_of_contents(contents: &[&[u8]]) -> String {
    let mut hasher = sha2::Sha256::new();
    for &content in contents {
        hasher = hasher.chain(content);
    }
    hasher.finalize().iter().map(|b| *b as char).collect()
}

pub fn get_sha256_of_file(path: &Path) -> Result<String, AgentError> {
    let content = read_content_of_file(path)?;
    Ok(get_sha256_of_contents(&[content.as_bytes()]))
}

pub fn read_content_of_file(path: &Path) -> Result<String, AgentError> {
    if !path.exists() {
        return Err(AgentError(format!("{:?} not exists", path)));
    }
    fs::read_to_string(path).map_err(|e| AgentError(format!("invalid UTF-8: {}", e)))
}

pub fn write_to_file(path: &Path, contents: &[&[u8]]) -> Result<(), AgentError> {
    write_or_append_to_file(path, contents, false)
}

pub fn append_to_file(path: &Path, contents: &[&[u8]]) -> Result<(), AgentError> {
    write_or_append_to_file(path, contents, true)
}

fn write_or_append_to_file(
    path: &Path,
    contents: &[&[u8]],
    is_append: bool,
) -> Result<(), AgentError> {
    let helper = || -> std::io::Result<()> {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .append(is_append)
            .open(path)?;

        for &content in contents {
            file.write_all(content)?;
        }
        file.flush()?;

        Ok(())
    };

    helper().map_err(to_agent_error)
}

pub fn to_agent_error<E: ToString>(e: E) -> AgentError {
    AgentError(e.to_string())
}

pub fn check_command_line(cmdline: &Vec<String>) -> Result<(), AgentError> {
    if cmdline.is_empty() {
        return Err(AgentError("allowed commands must not be empty".into()));
    }
    for arg in cmdline {
        let arg = arg.trim();
        if arg.is_empty() {
            return Err(AgentError("allowed commands must not be empty".into()));
        }
        if arg.contains('\0') {
            return Err(AgentError(
                "argv must contain non-empty strings without NUL".into(),
            ));
        }
    }
    Ok(())
}
