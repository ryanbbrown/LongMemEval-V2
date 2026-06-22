#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
METHOD="thinharness"
QUESTION_TYPE="dynamic-full"
DATA_ROOT_VALUE="${DATA_ROOT:-$REPO_ROOT/data/longmemeval-v2}"
OUTPUT_ROOT_VALUE="${OUTPUT_ROOT:-$REPO_ROOT/runs/thinharness_dynamic_items}"
TIER_VALUE="${TIER:-small}"
PYTHON_CMD=(uv run python)
READER_MODEL_VALUE="${READER_MODEL:-qwen/qwen3.5-9b}"
READER_BASE_URL_VALUE="${READER_BASE_URL:-https://openrouter.ai/api/v1}"
READER_API_KEY_ENV_VALUE="${READER_API_KEY_ENV:-OPENROUTER_API_KEY}"
MANAGER="$REPO_ROOT/evaluation/scripts/dynamic_full_run_manager.py"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "Missing OPENROUTER_API_KEY in environment or $REPO_ROOT/.env." >&2
  exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Missing OPENAI_API_KEY in environment or $REPO_ROOT/.env." >&2
  exit 1
fi

if [ "$#" -ne 0 ]; then
  echo "This script is an exact dynamic-full run manager and does not accept arguments." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT_VALUE"
"${PYTHON_CMD[@]}" "$MANAGER" \
  --data-root "$DATA_ROOT_VALUE" \
  --output-root "$OUTPUT_ROOT_VALUE"

run_missing_domain() {
  local domain="$1"
  local missing_file="$OUTPUT_ROOT_VALUE/${domain}_${QUESTION_TYPE}_missing_question_ids.txt"
  local ids=()
  while IFS= read -r question_id; do
    [ -n "$question_id" ] && ids+=("$question_id")
  done < "$missing_file"

  if [ "${#ids[@]}" -eq 0 ]; then
    echo "No missing $domain questions."
    return
  fi

  for question_id in "${ids[@]}"; do
    local output_dir="$OUTPUT_ROOT_VALUE/items/$domain/$question_id"
    echo "Running $domain question $question_id into $output_dir"
    rm -rf "$output_dir"
    env -u CODEX_API_KEY "${PYTHON_CMD[@]}" "$REPO_ROOT/evaluation/run_eval.py" \
      --method "$METHOD" \
      --data-root "$DATA_ROOT_VALUE" \
      --domain "$domain" \
      --tier "$TIER_VALUE" \
      --output-dir "$output_dir" \
      --reader-model "$READER_MODEL_VALUE" \
      --reader-base-url "$READER_BASE_URL_VALUE" \
      --reader-api-key-env "$READER_API_KEY_ENV_VALUE" \
      --thinharness-model "openai:gpt-5.4-mini" \
      --thinharness-api-key-env "OPENAI_API_KEY" \
      --thinharness-timeout-seconds "1800" \
      --thinharness-max-retries "3" \
      --thinharness-output-retries "1" \
      --thinharness-tool-retries "2" \
      --thinharness-reasoning-effort "xhigh" \
      --thinharness-tools read search jsonl_search list glob \
      --question-ids "$question_id"
  done
}

run_missing_domain web
run_missing_domain enterprise

"${PYTHON_CMD[@]}" "$MANAGER" \
  --data-root "$DATA_ROOT_VALUE" \
  --output-root "$OUTPUT_ROOT_VALUE"

echo "Done. Output root: $OUTPUT_ROOT_VALUE"
