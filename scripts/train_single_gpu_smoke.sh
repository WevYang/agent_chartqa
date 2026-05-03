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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

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
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mini_chartQA_smoke}"
MAX_STEPS="${MAX_STEPS:-1}"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    algorithm.kl_coef=0 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.max_prompt_length=4096 \
    data.max_response_length=128 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.padding_free=false \
    worker.actor.use_torch_compile=false \
    worker.critic.padding_free=false \
    worker.ref.padding_free=false \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=2 \
    worker.rollout.gpu_memory_utilization=0.35 \
    worker.rollout.max_model_len=5120 \
    worker.rollout.max_num_batched_tokens=5120 \
    worker.rollout.limit_images=1 \
    worker.rollout.enforce_eager=true \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.total_epochs=1 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_freq=-1 \
    trainer.logger='[console]' \
    worker.actor.global_batch_size=1 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.fsdp.enable_cpu_offload=true \
    data.rollout_batch_size=1 \
    data.val_batch_size=1 \
    trainer.save_checkpoint_path=./checkpoints/"${EXPERIMENT_NAME}" \
    worker.reward.reward_type=batch \
    worker.reward.reward_function=./examples/reward_function/refocus.py:compute_score
