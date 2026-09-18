# Rust Agent CLI and Python Model Service

`agent/cli.rs` connects the Rust learner agent to a standalone Python MLX-LM
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

## Tests

Rust tests create temporary directories under the repository's `tmp/`, using
`CARGO_MANIFEST_DIR` rather than the current directory or system `TMPDIR`.
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
