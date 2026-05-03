#!/bin/bash

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
TRAIN_FILES="${TRAIN_FILES:-datasets/train_full.parquet}"
VAL_FILES="${VAL_FILES:-datasets/val_full.parquet}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mini_chartQA}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
TP_SIZE="${TP_SIZE:-2}"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.rollout.tensor_parallel_size="${TP_SIZE}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS_PER_NODE}" \
    worker.actor.global_batch_size=8 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    data.rollout_batch_size=16 \
    data.val_batch_size=256 \
    trainer.save_checkpoint_path=./checkpoints/"${EXPERIMENT_NAME}" \
    worker.reward.reward_type=batch \
    worker.reward.reward_function=./examples/reward_function/refocus.py:compute_score
