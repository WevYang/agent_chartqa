#!/usr/bin/env bash
# run_baseline_eval.sh
#
# 1. (Optional) Convert v20 .pt checkpoint → HF safetensors via model_merger.py
# 2. Run zero-shot eval on the untrained Qwen2.5-VL-3B-Instruct (baseline)
# 3. Run single-pass eval on the fine-tuned v20 checkpoint
#
# Requirements: GPU with ≥8 GB VRAM, agent_chartqa_rt conda env.
# Usage:
#   bash scripts/run_baseline_eval.sh              # full run (baseline + v20)
#   bash scripts/run_baseline_eval.sh --baseline-only   # skip v20 eval
#   bash scripts/run_baseline_eval.sh --skip-merge      # skip model_merger if already done
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

PYTHON="${PYTHON:-python}"
VAL_FILE="${VAL_FILE:-datasets/val_small_128.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-docs/baseline_eval}"
BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
V20_CKPT_DIR="checkpoints/mini_chartQA_toolfixed_growthprompt_modelonly_n4_20step_v20/global_step_21/actor"
V20_HF_DIR="${V20_CKPT_DIR}/huggingface"

BASELINE_ONLY=0
SKIP_MERGE=0
for arg in "$@"; do
  [[ "$arg" == "--baseline-only" ]] && BASELINE_ONLY=1
  [[ "$arg" == "--skip-merge"    ]] && SKIP_MERGE=1
done

mkdir -p "$OUTPUT_DIR"

# ── Step 1: convert v20 .pt → HF safetensors ──────────────────────────────
if [[ "$BASELINE_ONLY" -eq 0 && "$SKIP_MERGE" -eq 0 ]]; then
  echo "=== [1/3] Converting v20 checkpoint to HF safetensors ==="
  # model_merger.py needs the actor dir (not the huggingface subdir)
  if ls "${V20_CKPT_DIR}"/model_world_size_*_rank_0.pt 1>/dev/null 2>&1; then
    $PYTHON scripts/model_merger.py --local_dir "$V20_CKPT_DIR"
    echo "Merge done. HF weights in: ${V20_HF_DIR}/"
  else
    echo "WARNING: no .pt weight found in ${V20_CKPT_DIR}/, skipping merge."
    SKIP_MERGE=1
  fi
else
  echo "=== [1/3] Skipping model merge (--skip-merge or --baseline-only) ==="
fi

# ── Step 2: zero-shot baseline ─────────────────────────────────────────────
echo ""
echo "=== [2/3] Zero-shot baseline: Qwen2.5-VL-3B-Instruct (untrained) ==="
$PYTHON scripts/run_baseline_eval.py \
  --model_path "$BASELINE_MODEL" \
  --val_file   "$VAL_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --label      baseline_zeroshot \
  --max_new_tokens 512 \
  --gpu_memory_utilization 0.70 \
  --max_model_len 4096

# ── Step 3: fine-tuned v20 single-pass eval ────────────────────────────────
if [[ "$BASELINE_ONLY" -eq 0 ]]; then
  echo ""
  echo "=== [3/3] Fine-tuned eval: v20 GRPO checkpoint (single-pass) ==="
  if ls "${V20_HF_DIR}"/*.safetensors 1>/dev/null 2>&1; then
    $PYTHON scripts/run_baseline_eval.py \
      --model_path "$V20_HF_DIR" \
      --val_file   "$VAL_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --label      v20_finetuned_singlepass \
      --max_new_tokens 512 \
      --gpu_memory_utilization 0.70 \
      --max_model_len 4096
  else
    echo "WARNING: no safetensors found in ${V20_HF_DIR}/ — v20 eval skipped."
    echo "Run without --skip-merge to convert the checkpoint first."
  fi
else
  echo "=== [3/3] Skipping v20 eval (--baseline-only) ==="
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=== Done. Results in ${OUTPUT_DIR}/ ==="
ls -lh "$OUTPUT_DIR/" 2>/dev/null || true
