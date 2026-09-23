use std::{
    fs,
    io::{Read, Write},
    path::Path,
    process::{Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use sha2::{Digest, digest::Update};

use crate::AgentError;

pub fn get_sha256_of_contents(contents: &[&[u8]]) -> String {
    let mut hasher = sha2::Sha256::new();
    for &content in contents {
        hasher = hasher.chain(content);
    }
    format!("{:x}", hasher.finalize())
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

pub fn run_command_with_timeout(
    mut cmdline: Command,
    timeout: Duration,
) -> Result<(Output, bool), AgentError> {
    let mut helper = || -> std::io::Result<(Output, bool)> {
        let mut child = cmdline
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let mut stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();

        let out_reader = thread::spawn(move || {
            let mut bytes = Vec::new();
            stdout.read_to_end(&mut bytes)?;
            Ok::<_, std::io::Error>(bytes)
        });
        let err_reader = thread::spawn(move || {
            let mut bytes = Vec::new();
            stderr.read_to_end(&mut bytes)?;
            Ok::<_, std::io::Error>(bytes)
        });

        let deadline = Instant::now() + timeout;
        let (status, timed_out) = loop {
            if let Some(status) = child.try_wait()? {
                break (status, false);
            }
            if Instant::now() >= deadline {
                child.kill()?;
                break (child.wait()?, true);
            }
            thread::sleep(Duration::from_millis(50));
        };

        let stdout = out_reader
            .join()
            .map_err(|_| std::io::Error::other("stdout reader panic"))??;
        let stderr = err_reader
            .join()
            .map_err(|_| std::io::Error::other("stderr reader panic"))??;

        Ok((
            Output {
                status,
                stdout,
                stderr,
            },
            timed_out,
        ))
    };

    helper().map_err(to_agent_error)
}
