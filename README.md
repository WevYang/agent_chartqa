# MINI_CHARTQA

一个面向 ChartQA 任务的训练复现仓库。当前代码基于 `verl` 训练框架，默认使用 `Qwen/Qwen2.5-VL-3B-Instruct` 作为基础模型，并通过自定义 reward function 对 ChartQA 数据进行训练。

## 项目内容

- `train.sh`：训练入口，已经改成可通过环境变量覆盖 GPU、模型路径和实验名
- `data/preprocess_data.sh`：下载 ChartQA 图片并生成 parquet 数据集
- `scripts/prepare_local.sh`：本地轻量准备脚本，适合在 GPU 开启前先执行
- `examples/config.yaml`：训练主配置
- `examples/reward_function/refocus.py`：评分函数
- `verl/`：训练、rollout、worker 等核心实现

## 运行环境

建议在 `orbit` 环境中执行：

```bash
conda activate orbit
cd ~/rivermind-data/agent_chartqa
```

当前依赖写在 `requirements.txt` 中，核心组件包括：

- `torch==2.5.1`
- `vllm==0.7.3`
- `xformers==0.0.28.post3`
- `accelerate==1.6.0`
- `pyarrow==22.0.0`
- `ray==2.40.0`
- `transformers==4.49.0`
- `datasets==3.2.0`
- `tensorboard==2.20.0`

安装依赖：

```bash
pip install -r requirements.txt
```

如果你的系统盘空间紧张，建议把运行时缓存统一落到 `~/rivermind-data`。当前 `train.sh` 和新增的单卡脚本会默认设置这些目录：

```bash
TMPDIR=~/rivermind-data/tmp
HF_HOME=~/rivermind-data/hf_home
TORCH_HOME=~/rivermind-data/torch_home
VLLM_CACHE_ROOT=~/rivermind-data/vllm_cache
WANDB_DIR=~/rivermind-data/wandb
SWANLAB_DIR=~/rivermind-data/swanlab
```

## 数据准备

仓库内已经包含：

- `data/train_chartqa_vcot.zip`

这个压缩包解开后会得到 `chartqa_vcot/train.jsonl`、`val.jsonl`、`test.jsonl`。真正训练还需要下载 ChartQA 图片包，脚本会自动处理。

### 1. 先做本地准备

这一步不依赖 GPU，可以先执行：

```bash
conda activate orbit
cd ~/rivermind-data/agent_chartqa
bash scripts/prepare_local.sh
```

这会：

- 创建 `checkpoints/` 和 `datasets/`
- 解压 `data/train_chartqa_vcot.zip`
- 如果你已经下载过 `data/ChartQA/`，会进一步生成 parquet 数据

### 2. 完整数据预处理

如果还没有 ChartQA 图片，执行：

```bash
conda activate orbit
cd ~/rivermind-data/agent_chartqa
bash data/preprocess_data.sh
```

这会：

- 下载 `ChartQA.zip`
- 解压图片目录到 `data/ChartQA/`
- 运行 `data/preprocess.py`
- 生成：
  - `datasets/train_full.parquet`
  - `datasets/val_full.parquet`

## 开始训练

默认训练命令：

```bash
conda activate orbit
cd ~/rivermind-data/agent_chartqa
bash train.sh
```

`train.sh` 默认参数：

- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct`
- `NUM_GPUS_PER_NODE=4`
- `TP_SIZE=2`
- `TRAIN_FILES=datasets/train_full.parquet`
- `VAL_FILES=datasets/val_full.parquet`
- `EXPERIMENT_NAME=mini_chartQA`

训练输出默认保存在：

```bash
checkpoints/mini_chartQA
```

如果你当前只有单卡，先做一次 smoke 验证更稳妥：

```bash
conda activate orbit
cd ~/rivermind-data/agent_chartqa
bash scripts/train_single_gpu_smoke.sh
```

这个脚本默认：

- 使用 `CUDA_VISIBLE_DEVICES=0`
- 使用 `tensor_parallel_size=1`
- 将 `trainer.max_steps` 设为 `1`
- 关闭训练前验证和周期性保存
- 仅保留 `console` logger，避免第一次跑链路时被外部日志平台卡住

## 常用覆盖方式

如果你想切换 GPU 或实验名，可以直接在命令前覆盖环境变量：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_GPUS_PER_NODE=4 \
TP_SIZE=2 \
EXPERIMENT_NAME=mini_chartQA_run2 \
bash train.sh
```

如果你想换基础模型：

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
bash train.sh
```

## 配置说明

主配置文件在 `examples/config.yaml`，当前训练逻辑的几个重点是：

- 算法使用 `grpo`
- actor 开启 gradient checkpointing
- rollout 默认 `tensor_parallel_size=2`
- reward function 使用 `examples/reward_function/refocus.py:compute_score`
- 训练日志配置为 `console` 和 `wandb`

注意：`examples/config.yaml` 里的默认模型和 GPU 数与 `train.sh` 中覆盖后的运行参数不完全一致。实际执行时以 `train.sh` 传入的参数为准。

另外，`datasets/*.parquet` 是本地生成产物，默认不建议提交到 GitHub；仓库中保留脚本和说明，数据按 README 步骤本地生成即可。

## 目录结构

```text
agent_chartqa/
├── data/
│   ├── preprocess.py
│   ├── preprocess_data.sh
│   └── train_chartqa_vcot.zip
├── datasets/
├── examples/
│   ├── config.yaml
│   ├── format_prompt/
│   └── reward_function/
├── judge/
├── scripts/
├── verl/
├── train.sh
└── requirements.txt
```

## 当前状态

目前仓库已经完成：

- 源码展开到仓库根目录
- 训练脚本参数化
- 数据准备脚本整理
- README 补全为可执行的复现说明

等 GPU 可用后，直接在 `orbit` 环境里继续执行数据预处理和训练即可。
