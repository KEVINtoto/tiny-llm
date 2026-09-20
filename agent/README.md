# Rust Agent CLI and Python Model Service

`src/cli.rs` connects the Rust learner agent to a standalone Python MLX-LM
service. Python owns model loading, chat-template rendering, and generation;
Rust sends the complete conversation on each request and owns the agent loop
and workspace tools. The original `agent.py` entrypoint is unchanged.


Run the commands below from the repository root.

Start the model service in one terminal:

```bash
pdm run model-service --model qwen3-4b --device gpu
```

The service defaults to `127.0.0.1:8000`. Use `--host` and `--port` to change
the address, or `--device cpu` to select CPU generation. `--model` accepts the
same shortcuts and full model names as `agent.py`. Model weights are loaded
once, before the server starts listening; an uncached model may be downloaded
by MLX-LM. Requests are processed serially.

After completing the required Rust exercises, run the CLI in another terminal
against a pre-created disposable workspace:

```bash
cargo run --bin agent-cli -- --root /path/to/workspace "Inspect README"
```

Use `--model-service-url http://127.0.0.1:8000` to select a different service.
The CLI retains `--max-steps` (8), `--max-tokens` (256),
`--enable-thinking`, `--allow-writes`, repeatable `--allow-command`, and
`--receipt-log` from `agent.py`. Model/device selection belongs to the Python
service. `--request-timeout` defaults to 300 seconds per generation request;
failed requests are not retried. Effect approvals require a TTY and an explicit
`y` or `yes`; enabling a tool alone does not approve an effect.

## Model API

`GET /health` returns the loaded model and generation device:

```json
{"status":"ok","model":"Qwen/Qwen3-4B-MLX-4bit","device":"gpu"}
```

`POST /generate` accepts a non-empty list of messages with string `role` and
`content` fields, a positive integer `max_tokens` (default 256), and a boolean
`enable_thinking` (default false):

```json
{
  "messages": [{"role":"user","content":"Reply with a short greeting."}],
  "max_tokens": 256,
  "enable_thinking": false
}
```

The response is `{"response":"the model's original text"}`. No trimming,
JSON action parsing, or thinking-tag removal happens in this transport.
Invalid requests return HTTP 400 and generation failures return HTTP 500,
both with `{"error":"description"}`. Unknown endpoints return HTTP 404.
The service has no conversation state, streaming, tools, authentication, or
dynamic model switching; it is intended for local use.

The service can be exercised before implementing the agent tools:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'
```

## Action JSON Schema

`ToolAction` is a Serde internally tagged enum: each tool has its own variant
with typed arguments. The JSON wire format stays flat:

```rust
use tiny_llm_agent::{ToolAction, action_schema};

let action: ToolAction =
    r#"{"tool":"read_file","path":"README.md"}"#.parse().unwrap();
assert_eq!(action, ToolAction::ReadFile { path: "README.md".into() });
let schema_json = serde_json::to_string_pretty(&action_schema()).unwrap();
```

`serde_json::from_str::<ToolAction>()` also accepts these JSON strings.
Schemars derives the schema from the same Serde definitions, including required
fields and `additionalProperties: false`. Missing arguments, unknown tools,
extra fields, and incorrect argument types are rejected. Only `list_files.path`
may be omitted; it defaults to `"."`. `run_command.argv` is an array of strings.
Final actions accept only `{"final":"..."}`; mixed final/tool objects are rejected.
`parse_action(response, Some(&available_tools))` additionally checks the runtime
tool allowlist, which is separate from the schema.

Run the focused protocol tests with `cargo test --lib protocol::tests`.

## Rust workspace

The repository root is a Cargo workspace with shared dependency versions:

`agent/Cargo.toml` defines the `tiny-llm-agent` library, `agent-cli` binary,
and integration tests. Source files live in `agent/src/`; Cargo automatically
discovers the test files in `agent/tests/`. Shared helpers live in
`tests/support/utils.rs` and are imported through `mod support` and
`use support::utils as test_utils`.

Run Cargo commands from the repository root. Existing test names are preserved,
for example `cargo test --test test_week_4_day_1`. Use
`cargo check --workspace --all-targets` to check the library, CLI, and tests.

## Tests

Rust tests create temporary directories under the repository's `tmp/`, using
the workspace member’s `CARGO_MANIFEST_DIR` to locate the repository root,
rather than the current directory or system `TMPDIR`.
Each test keeps its own isolated directory and removes it when finished.

The HTTP adapter tests are in [test_model_service.rs](tests/test_model_service.rs);
the Python service tests are in
[test_model_service.py](../tests_refsol/test_model_service.py).

Run the focused tests without downloading a model:

```bash
cargo test --bin agent-cli --test test_model_service
pdm run test-model-service
```

These tests use local fake HTTP services and injected model/tokenizer doubles;
they cover conversation transport, error handling, model reuse, generation
parameters, CLI validation, and approvals independently of the course TODOs.
