#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRACE_PATH="${1:-traces/mini_chartQA_toolfixed_growthprompt_modelonly_n4_20step_v20}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/agentic_eval/report.jsonl}"
MAX_RECORDS="${MAX_RECORDS:-256}"

if [[ -e "${TRACE_PATH}" ]]; then
  python -m agentic_eval.cli \
    --trace-path "${TRACE_PATH}" \
    --max-records "${MAX_RECORDS}" \
    --failures-only \
    --output "${OUTPUT_PATH}"
else
  python -m agentic_eval.cli \
    --demo \
    --failures-only \
    --output "${OUTPUT_PATH}"
fi
