//! CLI wiring for the learner agent and the standalone MLX model service.
//! Workspace/receipt/loop course TODOs are intentionally left to the learner.

#[cfg(test)]
#[path = "../tests/support/utils.rs"]
mod test_utils;

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use clap::{CommandFactory, Parser};
use serde_json::Value;
use tiny_llm_agent::model_service::HttpGenerator;
use tiny_llm_agent::protocol::AgentAction;
use tiny_llm_agent::workspace::ConfirmResult;
use tiny_llm_agent::{
    AgentError, AgentEvent, AgentLimits, ReceiptStore, ToolAction, ToolPolicy, Workspace, run_agent,
};

#[derive(Debug, Parser)]
#[command(
    about = "Run the Week 4 learner agent using a Python MLX model service.",
    after_help = "Requires completed workspace, receipt, and loop exercises. Existing course TODOs are not implemented by this CLI."
)]
struct Args {
    /// One natural-language goal
    #[arg(required = true, num_args = 1..)]
    task: Vec<String>,
    /// Pre-created disposable workspace containing no secrets
    #[arg(long)]
    root: PathBuf,
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    model_service_url: String,
    /// HTTP request timeout in seconds
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..))]
    request_timeout: u64,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(i64).range(1..))]
    max_steps: i64,
    #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(i64).range(1..))]
    max_tokens: i64,
    #[arg(long)]
    enable_thinking: bool,
    /// Enable approved write_file and edit_file calls
    #[arg(long)]
    allow_writes: bool,
    /// Allow one exact command; repeat to allow another
    #[arg(long, value_name = "COMMAND")]
    allow_command: Vec<String>,
    /// Append effect receipts to this workspace-relative JSONL path
    #[arg(long, value_name = "RELATIVE_PATH")]
    receipt_log: Option<PathBuf>,
}

struct Context {
    base_url: String,
    request_timeout: Duration,
    max_tokens: i64,
    enable_thinking: bool,

    task: String,
    workspace: Workspace,
    limits: AgentLimits,
    log_path: Option<PathBuf>,
}

impl Context {
    fn new(args: Args) -> Result<Self, AgentError> {
        let task = args.task.join(" ").trim().to_owned();
        if task.is_empty() {
            return Err(AgentError("task must not be empty".into()));
        }
        if args.max_steps <= 0 && args.max_tokens <= 0 {
            return Err(AgentError("generation limits must be positive".into()));
        }
        if !args.root.is_dir() {
            return Err(AgentError("--root must be a pre-created directory".into()));
        }

        let commands = parse_allowed_commands(&args.allow_command)?;

        let policy = ToolPolicy::new(
            args.root,
            ToolPolicy::DEFAULT_MAX_FILE_BYTES,
            ToolPolicy::DEFAULT_MAX_LIST_ENTRIES,
            args.allow_writes,
            commands,
            ToolPolicy::DEFAULT_MAX_WRITE_BYTES,
            ToolPolicy::DEFAULT_COMMAND_TIMEOUT_SECONDS,
        )?;

        let log_path = receipt_path(&policy.root, args.receipt_log.as_deref())?;

        let store = ReceiptStore::new(log_path.clone())?;

        let workspace = Workspace::new(policy, Some(Box::new(confirm_tool)), store);

        let limits = AgentLimits {
            max_steps: args.max_steps,
            ..AgentLimits::default()
        };

        Ok(Self {
            base_url: args.model_service_url,
            request_timeout: Duration::from_secs(args.request_timeout),
            max_tokens: args.max_tokens,
            enable_thinking: args.enable_thinking,
            task,
            workspace,
            limits,
            log_path,
        })
    }
}

fn parse_allowed_commands(values: &[String]) -> Result<Vec<Vec<String>>, AgentError> {
    values
        .iter()
        .map(|value| {
            // Python shlex.split defaults to comments=False, whereas Rust shlex
            // treats unquoted # as a comment. Protect it outside quotes first.
            let mut protected = String::new();
            let mut quote = None;
            let mut chars = value.chars();
            while let Some(ch) = chars.next() {
                if ch == '\\' && quote != Some('\'') {
                    protected.push(ch);
                    if let Some(next) = chars.next() {
                        protected.push(next);
                    }
                    continue;
                }
                if matches!(ch, '\'' | '"') {
                    if quote == Some(ch) {
                        quote = None;
                    } else if quote.is_none() {
                        quote = Some(ch);
                    }
                }
                if ch == '#' && quote.is_none() {
                    protected.push('\\');
                }
                protected.push(ch);
            }
            let command = shlex::split(&protected).ok_or_else(|| {
                AgentError("--allow-command contains invalid shell quoting".into())
            })?;
            if command.is_empty() {
                return Err(AgentError("--allow-command must not be empty".into()));
            }
            Ok(command)
        })
        .collect()
}

fn receipt_path(root: &Path, raw: Option<&Path>) -> Result<Option<PathBuf>, AgentError> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    if raw.is_absolute()
        || raw.file_name().is_none()
        || raw.components().any(|part| part == Component::ParentDir)
    {
        return Err(AgentError(
            "--receipt-log must be a workspace-relative file path".into(),
        ));
    }
    let path = root.join(raw);
    if !path.parent().is_some_and(Path::is_dir) {
        return Err(AgentError(
            "--receipt-log parent directory must exist".into(),
        ));
    }
    Ok(Some(path))
}

fn action_payload(action: &ToolAction) -> Value {
    serde_json::to_value(action).expect("tool actions must serialize")
}

fn confirm_with_io(
    action: &ToolAction,
    is_tty: bool,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> io::Result<bool> {
    writeln!(output, "\napproval requested> {}", action_payload(action))?;
    if action.tool() == "run_command" {
        writeln!(
            output,
            "warning> an allowed command is not confined by the workspace boundary"
        )?;
        writeln!(
            output,
            "warning> approved commands may create or change additional filesystem paths; command receipts record the command and result, not a complete filesystem diff"
        )?;
    }
    if !is_tty {
        writeln!(
            output,
            "approval denied> interactive confirmation requires a TTY"
        )?;
        return Ok(false);
    }
    write!(output, "approve this effect? [y/N] ")?;
    output.flush()?;
    let mut answer = String::new();
    if input.read_line(&mut answer)? == 0 {
        writeln!(output, "\napproval denied> no answer received")?;
        return Ok(false);
    }
    let approved = matches!(answer.trim().to_lowercase().as_str(), "y" | "yes");
    writeln!(output, "approval> {}", if approved { "yes" } else { "no" })?;
    Ok(approved)
}

fn confirm_tool(action: &ToolAction) -> ConfirmResult {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let approved = confirm_with_io(
        action,
        stdin.is_terminal(),
        &mut stdin.lock(),
        &mut stdout.lock(),
    )
    .unwrap_or(false);
    ConfirmResult::Approved(approved)
}

fn show_event(event: &AgentEvent) {
    println!("\n[{}] model> {}", event.step, event.response);
    if let Some(AgentAction::Tool(action)) = &event.action {
        println!("action> {}", action_payload(action));
    }
    if let Some(result) = &event.result {
        println!("observation> {result}");
    }
}

fn run(mut context: Context) -> Result<bool, AgentError> {
    let mut generate = HttpGenerator::new(
        &context.base_url,
        context.max_tokens,
        context.enable_thinking,
        context.request_timeout,
    )?;

    println!("\nmodel service> {}", context.base_url);
    println!("workspace> {}", context.workspace.policy.root.display());
    println!("goal> {}", context.task);
    let mut tools: Vec<_> = context
        .workspace
        .available_tools()
        .iter()
        .map(String::as_str)
        .collect();
    tools.sort_unstable();
    println!("tools> {}", tools.join(", "));
    if let Some(path) = context.log_path {
        println!("receipt log> {}", path.display());
    }
    let result = run_agent(
        Some(&context.task),
        &mut generate,
        &mut context.workspace,
        Some(&context.limits),
        Some(&show_event),
    )?;
    if result.completed {
        println!("\nfinished> {}", result.final_.as_deref().unwrap_or(""));
    } else {
        println!("\nstopped> {}", result.reason);
    }
    println!(
        "file-tool changes> {}",
        serde_json::json!(context.workspace.modified_files())
    );
    Ok(result.completed)
}

fn main() -> ExitCode {
    let args = Args::parse();
    let context = Context::new(args).unwrap_or_else(|e| {
        Args::command()
            .error(clap::error::ErrorKind::ValueValidation, e)
            .exit()
    });

    match run(context) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parser_validates_limits_and_required_arguments() {
        for option in ["--max-steps", "--max-tokens", "--request-timeout"] {
            for value in ["0", "-1", "abc"] {
                assert!(
                    Args::try_parse_from(["agent-cli", "--root", ".", option, value, "task"])
                        .is_err()
                );
            }
        }
        assert!(Args::try_parse_from(["agent-cli", "task"]).is_err());
        assert!(Args::try_parse_from(["agent-cli", "--root", "."]).is_err());
        assert_eq!(
            Args::try_parse_from(["agent-cli", "--help"])
                .unwrap_err()
                .kind(),
            clap::error::ErrorKind::DisplayHelp
        );
    }

    #[test]
    fn parses_task_and_defaults() {
        let root = test_utils::tempdir().unwrap();
        let path = root.path().to_str().unwrap();
        let args = Args::try_parse_from(["agent-cli", "--root", path, "检查", "README"]).unwrap();
        assert_eq!(args.task, ["检查", "README"]);
        assert_eq!(args.root, root.path());
        assert_eq!(
            (args.max_steps, args.max_tokens, args.request_timeout),
            (8, 256, 300)
        );
        assert!(!args.allow_writes && !args.enable_thinking);
        assert_eq!(args.model_service_url, "http://127.0.0.1:8000");
    }

    #[test]
    fn parses_explicit_feature_flags() {
        let args = Args::try_parse_from([
            "agent-cli",
            "--root",
            ".",
            "--allow-writes",
            "--enable-thinking",
            "hello",
        ])
        .unwrap();
        assert!(args.allow_writes);
        assert!(args.enable_thinking);
    }

    #[test]
    fn context_rejects_invalid_task_and_workspace_before_initializing_receipts() {
        let root = test_utils::tempdir().unwrap();
        let args =
            Args::try_parse_from(["agent-cli", "--root", root.path().to_str().unwrap(), "  "])
                .unwrap();
        let Err(error) = Context::new(args) else {
            panic!("an empty task must be rejected");
        };
        assert_eq!(error.0, "task must not be empty");

        let missing = root.path().join("missing");
        let args = Args::try_parse_from(["agent-cli", "--root", missing.to_str().unwrap(), "task"])
            .unwrap();
        let Err(error) = Context::new(args) else {
            panic!("a missing workspace must be rejected");
        };
        assert_eq!(error.0, "--root must be a pre-created directory");
    }

    #[test]
    fn commands_preserve_quotes_empty_arguments_and_literal_hashes() {
        let commands = parse_allowed_commands(&[
            r#"python -c 'print("hello world")'"#.into(),
            r##"echo "" a\ b #hash '#quoted' "#double" \#escaped"##.into(),
        ])
        .unwrap();
        assert_eq!(commands[0], ["python", "-c", "print(\"hello world\")"]);
        assert_eq!(
            commands[1],
            ["echo", "", "a b", "#hash", "#quoted", "#double", "#escaped"]
        );
        for value in [" ", "echo 'unterminated", "echo \\"] {
            assert!(parse_allowed_commands(&[value.into()]).is_err());
        }
    }

    #[test]
    fn receipt_log_requires_a_relative_path_and_existing_parent() {
        let root = test_utils::tempdir().unwrap();
        assert_eq!(receipt_path(root.path(), None).unwrap(), None);
        assert_eq!(
            receipt_path(root.path(), Some(Path::new("receipts.jsonl"))).unwrap(),
            Some(root.path().join("receipts.jsonl"))
        );
        for path in [
            "",
            ".",
            "..",
            "../outside.jsonl",
            "a/../receipt.jsonl",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tmp/receipt.jsonl"),
            "missing/log.jsonl",
        ] {
            assert!(
                receipt_path(root.path(), Some(Path::new(path))).is_err(),
                "{path}"
            );
        }
    }

    #[test]
    fn approval_requires_tty_and_explicit_yes() {
        let action = ToolAction::RunCommand {
            argv: vec!["echo".into(), "hello".into()],
        };
        let mut input = Cursor::new(b"yes\n");
        let mut output = Vec::new();
        assert!(!confirm_with_io(&action, false, &mut input, &mut output).unwrap());
        assert_eq!(input.position(), 0);
        assert!(
            String::from_utf8(output)
                .unwrap()
                .contains("requires a TTY")
        );
        for (answer, expected) in [
            (" Y \n", true),
            ("YES\n", true),
            ("n\n", false),
            ("\n", false),
            ("", false),
        ] {
            assert_eq!(
                confirm_with_io(&action, true, &mut Cursor::new(answer), &mut Vec::new()).unwrap(),
                expected
            );
        }
    }
}
