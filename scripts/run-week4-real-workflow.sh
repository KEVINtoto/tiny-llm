#!/usr/bin/env bash

# Reproduce the Week 4 real-model run using the cached Qwen3-4B model.
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

usage() {
    printf 'Usage: %s -d|--day <2|3> -s|--steps <default: 8> \n' "${0##*/}"
}

day=""
steps=8
while (( $# > 0 )); do
    case "$1" in
        --clear)
            rm -rf $repo_root/tmp/*
            exit 0
            ;;
        -s|--steps)
            steps="$2"
            shift 2
            ;;
        -d|--day)
            if [[ -n "$day" || $# -lt 2 ]]; then
                echo 'Provide -d|--day exactly once with a value of 2 or 3.' >&2
                usage >&2
                exit 2
            fi
            day="$2"
            case "$day" in
                2|3) ;;
                *)
                    printf 'Unsupported day: %s (expected 2 or 3).\n' "$day" >&2
                    exit 2
                    ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$day" ]]; then
    usage >&2
    exit 2
fi

run_root="$(mktemp -d $repo_root/tmp/tiny-llm-real-workflow-day$day-`date +%Y%m%d%H%M%S`)"
workspace="$run_root/workspace"

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

####################################################################################################

# Day 2 workflow
if [[ "$day" == 2 ]]; then

mkdir -p "$workspace/src"
printf '%s\n' '# Pocket Weather' 'A tiny terminal forecast project.' > "$workspace/README.md"
printf '%s\n' 'def forecast(city):' '    return f"Sunny in {city}"' > "$workspace/src/weather.py"
printf 'Run directory: %s\n' "$run_root"

task='Inspect this workspace and explain its purpose and the behavior implemented in its source file. Use the available workspace tools to gather evidence from both the project overview and the source file. Your first response must be one tool request, every response must contain exactly one JSON object, and you must not finish until you have read the source file.'

# Retain the CLI's exit code, including a TODO panic, and still verify files.
set +e
cargo run agent-cli --root "$workspace" --max-steps "$steps" "$task" 2>&1 | tee "$run_root/workflow.log"
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

fi
# End of Day 2 workflow

####################################################################################################

# Day 3 workflow
if [[ "$day" == 3 ]]; then

mkdir -p "$workspace"
printf '%s\n' 'def greeting(name):' '    return f"Hello, {name}!"' > "$workspace/app.py"
cat > "$workspace/validate.py" <<'PYTHON'
from pathlib import Path
from app import greeting

assert greeting("Ada") == "Welcome, Ada!"
assert Path("NOTES.md").read_text() == "Greeting now says Welcome.\n"
print("validation passed")
PYTHON
printf 'Run directory: %s\n' "$run_root"

task="Inspect the workspace. Create NOTES.md containing exactly 'Greeting now says Welcome.' followed by a newline, precisely change app.py so greeting says Welcome instead of Hello, run the allowed validation, react to its evidence, and finish with a brief summary. Use workspace tools to gather evidence before any effect. Your first response must be one tool request, every response must contain exactly one JSON object, and do not finish until the requested files and validation evidence have been inspected."

printf '%s\n' 'Read each approval payload before answering y; the default is No.'

# Retain the CLI's exit code, including a TODO panic, and still inspect files and receipts.
set +e
cargo run agent-cli --root "$workspace" \
    --allow-writes \
    --allow-command "python validate.py" \
    --receipt-log .agent-receipts.jsonl \
    --max-steps "$steps" \
    "$task" 2>&1 | tee "$run_root/workflow.log"
cli_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$cli_status" > "$run_root/exit-code.txt"
printf 'CLI exit code: %s\n' "$cli_status"

# Show evidence even if the model stopped early or an approval was denied.
# Do not rerun validation here: its recorded tool result is the live-run evidence.
for artifact in app.py NOTES.md .agent-receipts.jsonl; do
    printf '\n--- %s ---\n' "$artifact"
    if [[ -f "$workspace/$artifact" ]]; then
        cat "$workspace/$artifact"
        printf '\n'
    else
        printf 'Not created: %s\n' "$workspace/$artifact"
    fi
done

printf 'Compare the changed files with the validation status and output in the receipts.\n'
printf 'Workflow log: %s/workflow.log\nService log: %s/model-service.log\n' "$run_root" "$run_root"
exit "$cli_status"

fi
# End of Day 3 workflow