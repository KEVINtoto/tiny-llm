//! Reusable helpers for the Week 4 integration tests.
// Each test target uses a different subset of this module.
#![allow(dead_code, unused_imports)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use tiny_llm_agent::generation::Message;
use tiny_llm_agent::protocol::AgentAction;
use tiny_llm_agent::{
    AgentError, AgentLimits, FileExpectation, ReceiptExpectation, ReceiptStore, ResultExpectation,
    ToolAction, ToolPolicy, Workspace,
};

pub fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("agent package must be under the repository root")
        .to_path_buf()
}

pub fn root() -> PathBuf {
    let root = repository_root().join("tmp");
    std::fs::create_dir_all(&root).expect("create project temporary directory");
    root
}

pub fn tempdir() -> std::io::Result<tempfile::TempDir> {
    tempfile::Builder::new()
        .prefix("tiny-llm-agent-")
        .tempdir_in(root())
}

pub fn json_arguments(value: Value) -> Map<String, Value> {
    value
        .as_object()
        .expect("test action arguments must be a JSON object")
        .clone()
}

pub fn tool_action(tool: &str, arguments: Value) -> ToolAction {
    let mut payload = json_arguments(arguments);
    payload.insert("tool".into(), Value::String(tool.to_owned()));
    serde_json::from_value(Value::Object(payload)).expect("valid tool action")
}

pub fn message(role: &str, content: impl Into<String>) -> Message {
    Message::from([
        ("role".into(), role.into()),
        ("content".into(), content.into()),
    ])
}

/// Policy for fake workspaces, without filesystem validation.
pub fn policy_with_tools(root: &str) -> ToolPolicy {
    ToolPolicy {
        root: PathBuf::from(root),
        max_file_bytes: ToolPolicy::DEFAULT_MAX_FILE_BYTES,
        max_list_entries: ToolPolicy::DEFAULT_MAX_LIST_ENTRIES,
        allow_writes: false,
        allowed_commands: Vec::new(),
        max_write_bytes: ToolPolicy::DEFAULT_MAX_WRITE_BYTES,
        command_timeout_seconds: ToolPolicy::DEFAULT_COMMAND_TIMEOUT_SECONDS,
    }
}

pub fn scripted_responses(items: &[&str]) -> impl Fn(&[Message]) -> String {
    let queue = RefCell::new(
        items
            .iter()
            .map(|item| (*item).to_owned())
            .collect::<VecDeque<_>>(),
    );
    move |_messages| {
        queue
            .borrow_mut()
            .pop_front()
            .expect("scripted response queue exhausted")
    }
}

pub fn limits(
    max_steps: i64,
    max_context_chars: i64,
    max_invalid_actions: i64,
    max_identical_actions: i64,
) -> AgentLimits {
    AgentLimits::new(
        max_steps,
        max_context_chars,
        max_invalid_actions,
        max_identical_actions,
    )
    .expect("test limits must be valid")
}

pub fn make_bounded_workspace(
    root: &Path,
    max_file_bytes: i64,
    max_list_entries: i64,
) -> Workspace {
    Workspace::new(
        bounded_policy(root, max_file_bytes, max_list_entries),
        None,
        ReceiptStore::new(None).unwrap(),
    )
}

pub fn policy(root: &Path, allow_writes: bool, allowed_commands: Vec<Vec<String>>) -> ToolPolicy {
    ToolPolicy::new(
        root.to_path_buf(),
        ToolPolicy::DEFAULT_MAX_FILE_BYTES,
        ToolPolicy::DEFAULT_MAX_LIST_ENTRIES,
        allow_writes,
        allowed_commands,
        ToolPolicy::DEFAULT_MAX_WRITE_BYTES,
        ToolPolicy::DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    .unwrap()
}

pub fn memory_store() -> ReceiptStore {
    ReceiptStore::new(None).unwrap()
}

pub fn make_workspace(root: &Path) -> Workspace {
    Workspace::new(policy(root, false, Vec::new()), None, memory_store())
}

pub fn read_only_workspace(root: &Path) -> Workspace {
    Workspace::new(policy(root, false, Vec::new()), None, memory_store())
}

pub fn string_vec(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| (*part).to_owned()).collect()
}

pub fn assert_error_contains<T: std::fmt::Debug>(result: Result<T, AgentError>, expected: &str) {
    let error = result.expect_err("operation unexpectedly succeeded");
    assert!(
        error.to_string().contains(expected),
        "expected error containing {expected:?}, got {error}"
    );
}

pub fn count_fake_tokens(messages: &[Message]) -> i64 {
    messages
        .iter()
        .map(|message| message["content"].chars().count() as i64)
        .sum()
}

pub fn file_expectation(path: &str, content: &str) -> FileExpectation {
    FileExpectation::new(path.to_owned(), content.to_owned()).expect("valid file expectation")
}

pub fn result_expectation(tool: &str, contains: &str) -> ResultExpectation {
    ResultExpectation::new(tool.to_owned(), contains.to_owned()).expect("valid result expectation")
}

pub fn receipt_expectation(
    call_id: &str,
    tool: &str,
    exit_state: &str,
    contains: &str,
    changed: &[&str],
) -> ReceiptExpectation {
    ReceiptExpectation::new(
        call_id.to_owned(),
        tool.to_owned(),
        exit_state.to_owned(),
        contains.to_owned(),
        changed.iter().map(|path| (*path).to_owned()).collect(),
    )
    .expect("valid receipt expectation")
}

pub fn action_tool(event: &tiny_llm_agent::AgentEvent) -> Option<&str> {
    match event.action.as_ref() {
        Some(AgentAction::Tool(action)) => Some(action.tool()),
        _ => None,
    }
}

pub fn failed_names(report: &tiny_llm_agent::EvaluationReport) -> Vec<&str> {
    report
        .checks
        .iter()
        .filter(|check| !check.passed)
        .map(|check| check.name.as_str())
        .collect()
}

pub fn payload(observation: &str, prefix: &str) -> Value {
    let encoded = observation
        .strip_prefix(prefix)
        .unwrap_or_else(|| panic!("observation did not start with {prefix:?}: {observation:?}"));
    serde_json::from_str(encoded).expect("observation payload is valid JSON")
}

pub fn sha256(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

pub fn bounded_policy(root: &Path, max_file_bytes: i64, max_list_entries: i64) -> ToolPolicy {
    ToolPolicy::new(
        root.to_path_buf(),
        max_file_bytes,
        max_list_entries,
        false,
        Vec::new(),
        ToolPolicy::DEFAULT_MAX_WRITE_BYTES,
        ToolPolicy::DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    .unwrap()
}

/// One deterministic fake token per message, encoding its character count.
pub fn fake_token_ids(messages: &[Message]) -> Vec<i64> {
    messages
        .iter()
        .map(|message| message["content"].chars().count() as i64)
        .collect()
}

/// An isolated directory under the checkout, removed when the test releases it.
pub struct TestDir(tempfile::TempDir);

impl TestDir {
    pub fn new(label: &str) -> Self {
        Self(
            tempfile::Builder::new()
                .prefix(&format!("tiny-llm-{label}-"))
                .tempdir_in(root())
                .expect("create isolated test directory"),
        )
    }

    pub fn path(&self) -> &Path {
        self.0.path()
    }
}

pub fn list_files_action(path: &str) -> ToolAction {
    ToolAction::ListFiles { path: path.into() }
}
pub fn read_file_action(path: &str) -> ToolAction {
    ToolAction::ReadFile { path: path.into() }
}
pub fn write_file_action(path: &str, content: &str) -> ToolAction {
    ToolAction::WriteFile {
        path: path.into(),
        content: content.into(),
    }
}
pub fn edit_file_action(path: &str, old: &str, new: &str) -> ToolAction {
    ToolAction::EditFile {
        path: path.into(),
        old: old.into(),
        new: new.into(),
    }
}
pub fn run_command_action(argv: Vec<String>) -> ToolAction {
    ToolAction::RunCommand { argv }
}
