#!/usr/bin/env bash
# Reproduce the Week 4 Day 2 real-model run using the cached Qwen3-4B model.
# Logs and the disposable workspace are retained in the printed run directory.
# This intentionally uses the CLI's default 8-step budget. 
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python="$repo_root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Missing .venv/bin/python; install the project dependencies first." >&2
    exit 1
fi

run_root="$(mktemp -d $repo_root/tmp/tiny-llm-real-workflow-XXXXXX)"
workspace="$run_root/workspace"
mkdir -p "$workspace/src"
printf '%s\n' '# Pocket Weather' 'A tiny terminal forecast project.' > "$workspace/README.md"
printf '%s\n' 'def forecast(city):' '    return f"Sunny in {city}"' > "$workspace/src/weather.py"
printf 'Run directory: %s\n' "$run_root"

service_pid=""
cleanup() {
    if [[ -n "$service_pid" ]]; then
        kill "$service_pid" 2>/dev/null || true
        wait "$service_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Use cached weights only, without downloading a different model.
HF_HUB_OFFLINE=1 "$python" -u model_service.py \
    --model qwen3-4b --device gpu --port 8000 \
    > "$run_root/model-service.log" 2>&1 &
service_pid=$!

# Wait for model loading and verify the actual service before running the CLI.
service_ready=false
for ((attempt = 0; attempt < 120; attempt++)); do
    if ! kill -0 "$service_pid" 2>/dev/null; then
        cat "$run_root/model-service.log" >&2
        exit 1
    fi
    if curl --noproxy '*' --fail --silent --max-time 2 \
        http://127.0.0.1:8000/health > "$run_root/health.json"; then
        service_ready=true
        break
    fi
    sleep 1
done
if [[ "$service_ready" != true ]]; then
    echo "Model service did not become ready; see $run_root/model-service.log" >&2
    exit 1
fi
cat "$run_root/health.json"
printf '\n'

task='Inspect this workspace and explain its purpose and the behavior implemented in its source file. Use the available workspace tools to gather evidence from both the project overview and the source file. Your first response must be one tool request, every response must contain exactly one JSON object, and you must not finish until you have read the source file.'

# Retain the CLI's exit code, including a TODO panic, and still verify files.
set +e
cargo run agent-cli --root "$workspace" "$task" 2>&1 | tee "$run_root/workflow.log"
cli_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$cli_status" > "$run_root/exit-code.txt"
printf 'CLI exit code: %s\n' "$cli_status"

"$python" - "$workspace" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {
    "README.md": b"# Pocket Weather\nA tiny terminal forecast project.\n",
    "src/weather.py": b'def forecast(city):\n    return f"Sunny in {city}"\n',
}
actual = {
    str(path.relative_to(root)): path.read_bytes()
    for path in root.rglob("*") if path.is_file()
}
assert actual == expected, actual
print("Verified: both workspace files are unchanged; no additional files were created.")
PY

printf 'Workflow log: %s/workflow.log\nService log: %s/model-service.log\n' "$run_root" "$run_root"
exit "$cli_status"
