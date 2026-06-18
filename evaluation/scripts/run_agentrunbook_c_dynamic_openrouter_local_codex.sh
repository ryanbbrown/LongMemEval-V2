#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
METHOD="agentrunbook_c"
QUESTION_TYPE="dynamic-environment"
DATA_ROOT_VALUE="$REPO_ROOT/data/longmemeval-v2"
OUTPUT_ROOT_VALUE="$REPO_ROOT/runs/agentrunbook_c_dynamic_openrouter_local_codex_xhigh"
TIER_VALUE="small"
PYTHON_CMD=(uv run python)
CODEX_BINARY_VALUE="codex"
CODEX_MODEL_VALUE="gpt-5.4-mini"
CODEX_REASONING_EFFORT_VALUE="xhigh"
READER_MODEL_VALUE="qwen/qwen3.5-9b"
READER_BASE_URL_VALUE="https://openrouter.ai/api/v1"
READER_API_KEY_ENV_VALUE="OPENROUTER_API_KEY"
WEB_QUESTION_IDS=(
  00aa905a 06a5a25f 07d86be1 11dac74b 17d63ad8 1defc293 2ee130d2 4329b535
  5387d23d 5a85ca0f 5c826b28 5cb7289b 609acb91 63a1b94b 65fbcbdf 6652c337
  691cd1da 6a23675f 6a5bbcdb 6a73e32b 6f9cb5fc 744dbdad 820d56e2 8955b988
  8cb3280c 90dc7e66 91d0775e 96497069 a12b7094 b2595523 b5a25676 b828a6b2
  bdf0e84b c36b0d68 c738b934 c843c4ec d3300354 dae9f7e9 dea446d0 deb008ed
  e71f7c92 e83245c5 e9bb843a eab07e81 edb69441 edea0219 ee68431d f93031d2
  f9f9cd61 fe18a443 ffa4e57e
)
ENTERPRISE_QUESTION_IDS=(
  01307e07 0d07143c 14a823df 233f9f09 2721ca7f 2d4e08b9 32d04f31 3fd81278
  41a18e5c 4c6d1eb7 4dffe641 50711c52 5edd2533 658fa827 6cb8ce37 78686f4e
  7ea13f14 82cdfcc3 85fc5722 876c720a 87711b62 882965b7 901b7d17 984022b1
  9c24f41e aa64a8bd b54161f8 b8cabd09 bdb825a3 c0696888 d0d61088 d83c4bf7
  dd53ab77 e033e796 f35bac4a
)

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "Missing OPENROUTER_API_KEY in environment or $REPO_ROOT/.env." >&2
  exit 1
fi

if [ "$#" -ne 0 ]; then
  echo "This script is an exact run record and does not accept arguments." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT_VALUE"

printf "%s\n" "${WEB_QUESTION_IDS[@]}" > "$OUTPUT_ROOT_VALUE/web_${QUESTION_TYPE}_question_ids.txt"
printf "%s\n" "${ENTERPRISE_QUESTION_IDS[@]}" > "$OUTPUT_ROOT_VALUE/enterprise_${QUESTION_TYPE}_question_ids.txt"

env -u OPENAI_API_KEY -u CODEX_API_KEY "${PYTHON_CMD[@]}" "$REPO_ROOT/evaluation/run_eval.py" \
  --method "$METHOD" \
  --data-root "$DATA_ROOT_VALUE" \
  --domain web \
  --tier "$TIER_VALUE" \
  --output-dir "$OUTPUT_ROOT_VALUE/${METHOD}_${QUESTION_TYPE}_web_${TIER_VALUE}" \
  --codex-binary "$CODEX_BINARY_VALUE" \
  --codex-model "$CODEX_MODEL_VALUE" \
  --codex-reasoning-effort "$CODEX_REASONING_EFFORT_VALUE" \
  --reader-model "$READER_MODEL_VALUE" \
  --reader-base-url "$READER_BASE_URL_VALUE" \
  --reader-api-key-env "$READER_API_KEY_ENV_VALUE" \
  --question-ids "${WEB_QUESTION_IDS[@]}"

env -u OPENAI_API_KEY -u CODEX_API_KEY "${PYTHON_CMD[@]}" "$REPO_ROOT/evaluation/run_eval.py" \
  --method "$METHOD" \
  --data-root "$DATA_ROOT_VALUE" \
  --domain enterprise \
  --tier "$TIER_VALUE" \
  --output-dir "$OUTPUT_ROOT_VALUE/${METHOD}_${QUESTION_TYPE}_enterprise_${TIER_VALUE}" \
  --codex-binary "$CODEX_BINARY_VALUE" \
  --codex-model "$CODEX_MODEL_VALUE" \
  --codex-reasoning-effort "$CODEX_REASONING_EFFORT_VALUE" \
  --reader-model "$READER_MODEL_VALUE" \
  --reader-base-url "$READER_BASE_URL_VALUE" \
  --reader-api-key-env "$READER_API_KEY_ENV_VALUE" \
  --question-ids "${ENTERPRISE_QUESTION_IDS[@]}"

"${PYTHON_CMD[@]}" "$REPO_ROOT/leaderboard/combine_aggregated_metrics.py" \
  "$OUTPUT_ROOT_VALUE/${METHOD}_${QUESTION_TYPE}_web_${TIER_VALUE}/aggregated_metrics.json" \
  "$OUTPUT_ROOT_VALUE/${METHOD}_${QUESTION_TYPE}_enterprise_${TIER_VALUE}/aggregated_metrics.json" \
  -o "$OUTPUT_ROOT_VALUE/${METHOD}_${QUESTION_TYPE}_combined_${TIER_VALUE}_metrics.json"
