#!/bin/bash

set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
export RIVERMIND_DATA_ROOT="${RIVERMIND_DATA_ROOT:-$HOME/rivermind-data}"
export TMPDIR="${TMPDIR:-$RIVERMIND_DATA_ROOT/tmp}"
export HF_HOME="${HF_HOME:-$RIVERMIND_DATA_ROOT/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$RIVERMIND_DATA_ROOT/torch_home}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$RIVERMIND_DATA_ROOT/vllm_cache}"
export WANDB_DIR="${WANDB_DIR:-$RIVERMIND_DATA_ROOT/wandb}"
export SWANLAB_DIR="${SWANLAB_DIR:-$RIVERMIND_DATA_ROOT/swanlab}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RIVERMIND_DATA_ROOT/.cache}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export VERL_SKIP_POST_UPDATE_OFFLOAD="${VERL_SKIP_POST_UPDATE_OFFLOAD:-0}"
export VERL_ADAMW_FUSED="${VERL_ADAMW_FUSED:-0}"
export VERL_ANYPRECISION_USE_KAHAN="${VERL_ANYPRECISION_USE_KAHAN:-0}"
export VERL_DISABLE_VALIDATION="${VERL_DISABLE_VALIDATION:-1}"
export VERL_DISABLE_CHECKPOINT_SAVE="${VERL_DISABLE_CHECKPOINT_SAVE:-1}"
ACTOR_OFFLOAD_OPTIMIZER="${ACTOR_OFFLOAD_OPTIMIZER:-false}"
VAL_FREQ="${VAL_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
SAVE_FREQ="${SAVE_FREQ:--1}"
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-false}"
ROLLOUT_LIMIT_IMAGES="${ROLLOUT_LIMIT_IMAGES:-1}"
TRACE_ENABLE="${TRACE_ENABLE:-false}"
TRACE_OUTPUT_DIR="${TRACE_OUTPUT_DIR:-}"
TRACE_SAMPLE_SIZE="${TRACE_SAMPLE_SIZE:-64}"
TRACE_SAVE_IMAGES="${TRACE_SAVE_IMAGES:-true}"
TRACE_MAX_STEPS="${TRACE_MAX_STEPS:-200}"

mkdir -p \
  "$TMPDIR" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$WANDB_DIR" \
  "$SWANLAB_DIR" \
  "$XDG_CACHE_HOME"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
TRAIN_FILES="${TRAIN_FILES:-datasets/train_full.parquet}"
VAL_FILES="${VAL_FILES:-datasets/val_full.parquet}"
FORMAT_PROMPT="${FORMAT_PROMPT:-examples/format_prompt/chartQA.jinja}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mini_chartQA_smoke}"
MAX_STEPS="${MAX_STEPS:-1}"
DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-4096}"
DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-128}"
DATA_MAX_PIXELS="${DATA_MAX_PIXELS:-4194304}"
DATA_MIN_PIXELS="${DATA_MIN_PIXELS:-262144}"
ROLLOUT_N="${ROLLOUT_N:-2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-5120}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-5120}"
ACTOR_MICRO_BATCH_SIZE_UPDATE="${ACTOR_MICRO_BATCH_SIZE_UPDATE:-2}"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    algorithm.kl_coef=0 \
    algorithm.online_filtering=false \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.max_prompt_length="${DATA_MAX_PROMPT_LENGTH}" \
    data.max_response_length="${DATA_MAX_RESPONSE_LENGTH}" \
    data.max_pixels="${DATA_MAX_PIXELS}" \
    data.min_pixels="${DATA_MIN_PIXELS}" \
    data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS}" \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.padding_free=false \
    worker.actor.use_torch_compile=false \
    worker.critic.padding_free=false \
    worker.ref.padding_free=false \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n="${ROLLOUT_N}" \
    worker.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    worker.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    worker.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    worker.rollout.limit_images="${ROLLOUT_LIMIT_IMAGES}" \
    worker.rollout.enforce_eager=true \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.total_epochs=1 \
    trainer.val_freq="${VAL_FREQ}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.logger='[console]' \
    trainer.trace.enable="${TRACE_ENABLE}" \
    trainer.trace.output_dir="${TRACE_OUTPUT_DIR}" \
    trainer.trace.sample_size_per_step="${TRACE_SAMPLE_SIZE}" \
    trainer.trace.save_images="${TRACE_SAVE_IMAGES}" \
    trainer.trace.max_steps="${TRACE_MAX_STEPS}" \
    worker.actor.global_batch_size=1 \
    worker.actor.micro_batch_size_per_device_for_update="${ACTOR_MICRO_BATCH_SIZE_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.fsdp.enable_cpu_offload=false \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer="${ACTOR_OFFLOAD_OPTIMIZER}" \
    data.rollout_batch_size=1 \
    data.val_batch_size=1 \
    trainer.save_checkpoint_path=./checkpoints/"${EXPERIMENT_NAME}" \
    worker.reward.reward_type=llm_batch \
    worker.reward.reward_function=./examples/reward_function/refocus.py:compute_score
