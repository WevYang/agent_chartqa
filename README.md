# Agent-ChartQA

基于 `Qwen/Qwen2.5-VL-3B-Instruct`、`veRL`、`vLLM` 和 `PyTorch/FSDP` 的 ChartQA 多模态图表问答强化学习复现项目。项目重点不是单纯跑通脚本，而是补齐了 ChartQA 数据预处理、bbox-aware 图表聚焦工具、工具调用奖励、GRPO 训练、trace 归因、model-only checkpoint 保存和 Hugging Face 发布闭环。

## 当前产物

- GitHub 源码仓库：`https://github.com/WevYang/agent_chartqa`
- Hugging Face checkpoint：`https://huggingface.co/AuroraVvvv/qwen2.5_instruct_3b_rl`
- HF 最新校验 commit：`d842c44425b14072a1667e34e5e0aa0aed63d775`
- 基础模型：`Qwen/Qwen2.5-VL-3B-Instruct`
- 训练数据：ChartQA / ChartQA-vcot 风格样本，训练前转换为 parquet
- 训练算法：GRPO，单卡 FSDP/NO_SHARD 复现
- 最终 checkpoint：`mini_chartQA_toolfixed_growthprompt_modelonly_n4_20step_v20/global_step_21/actor`

HF 仓库中已上传：

| 文件 | 说明 |
| --- | --- |
| `model_world_size_1_rank_0.pt` | model-only PyTorch state dict，大小 `16,263,431,710` bytes，LFS 存储 |
| `extra_state_world_size_1_rank_0.pt` | veRL/FSDP 额外状态 |
| `huggingface/` | tokenizer、config、preprocessor 等基础模型配置文件 |
| `README.md` | HF model card |

注意：当前 HF 产物是 veRL/FSDP 保存出的 model-only state dict，不是标准 `save_pretrained()` 目录，因此不能直接用 `AutoModel.from_pretrained()` 加载。若要做标准推理发布，需要再转换为 Hugging Face `safetensors` 或 `pytorch_model.bin` 分片格式。

## 核心改造

- 数据链路：将 ChartQA 图像、问题、答案和 `x_values_bbox`、`y_values_bbox` 等结构化定位信息整理为 parquet 多模态训练样本。
- 工具调用：实现并修复 bbox-aware 图表聚焦工具，支持对图表 x/y 标签做大小写、单复数、星号后缀和 token-subset 模糊匹配。
- Prompt 约束：新增 `examples/format_prompt/chartQA_short.jinja`，强制模型先调用 focus 工具，再根据观察结果输出 `FINAL ANSWER`。
- 奖励函数：新增 `examples/reward_function/refocus.py`，包含格式奖励、工具调用奖励、relaxed numeric matching、`FINAL ANSWER`/`ANSWER`/`\boxed{}` 鲁棒解析。
- 训练稳定性：修复 vLLM rollout 进度条噪声、trace 图像序列化、工具执行奖励归因和单卡 FSDP checkpoint 保存问题。
- 显存与磁盘：通过 offload、micro-batch、像素裁剪、response length 控制和 model-only checkpoint，解决 48GB 单卡训练与磁盘空间不足问题。

## 实验结果

以下结果来自本仓库训练过程中的内部 reward/trace 评估，不是官方 ChartQA test split accuracy。官方测试准确率需要单独跑标准 ChartQA evaluator。

| 实验 | 设置 | `reward/overall` | `reward/accuracy` | 工具/格式 | 备注 |
| --- | --- | --- | --- | --- | --- |
| v18 | 5-step quick run | mean `0.9180` | mean `0.8500` | `1.0 / 1.0` | 快速确认工具奖励链路 |
| v19 | 20-step, balanced prompt, `rollout_n=4` | mean `0.9163`, first10 `0.9020`, last10 `0.9305` | mean `0.8478`, last10 `0.8737` | `1.0 / 1.0` | 无 response clipping |
| v20 | 20-step, growth prompt, model-only save, `rollout_n=4` | mean `0.9204`, first10 `0.9037`, last10 `0.9371` | mean `0.8553`, first10 `0.8250`, last10 `0.8857` | `1.0 / 1.0` | 最终上传 checkpoint |

v20 结论：

- `reward/tool` 和 `reward/format` 全程保持 `1.0`，说明模型稳定学会先调用图表聚焦工具，再按指定格式输出答案。
- `reward/overall` 后 10 步提升到 `0.9371`，优于 v19 的 `0.9305`。
- `reward/accuracy` 后 10 步达到 `0.8857`，相比前 10 步 `0.8250` 有明显提升。
- 相比 v19，v20 的 `reward/overall` 均值从 `0.9163` 提升到 `0.9204`，后半程从 `0.9305` 提升到 `0.9371`；`reward/accuracy` 均值从 `0.8478` 提升到 `0.8554`，后半程从 `0.8737` 提升到 `0.8857`。
- `response_clip_ratio=0`，说明当前 response length 设置没有造成答案截断。
- 平均 response length 约 `45.6` tokens，step 14 出现过 `102.25` 的长推理链，说明模型会在部分复杂题上生成更完整的推理。

运行资源：

- 单张 RTX 4090 48GB
- v20 20-step 训练平均单步约 `34s`
- v20 token 吞吐约 `215 tok/s`
- 显存峰值约 `46.6GB`
- 完整 optimizer checkpoint 未保存，最终采用 model-only checkpoint，避免磁盘被 FSDP optimizer state 占满

已知误差来源：

- ranking 类问题：模型有时输出数值而不是类别名。
- between/range 类问题：模型容易漏掉区间边界或只聚焦一个标签。
- difference/growth 类问题：模型可能只读取起点或终点，导致差值/增长率错误。
- 当前结果是内部训练 reward，不应直接写成 ChartQA 官方 test accuracy。

## 运行环境

建议使用本地 conda 环境执行：

```bash
conda activate agent_chartqa_rt
cd ~/rivermind-data/agent_chartqa
```

核心依赖见 `requirements.txt`，包括：

- `torch==2.5.1`
- `vllm==0.7.3`
- `xformers==0.0.28.post3`
- `accelerate==1.6.0`
- `pyarrow==22.0.0`
- `ray==2.40.0`
- `transformers==4.49.0`
- `datasets==3.2.0`
- `tensorboard==2.20.0`

安装：

```bash
pip install -r requirements.txt
```

系统盘空间不足时，建议把缓存统一放到 `~/rivermind-data`：

```bash
export TMPDIR=~/rivermind-data/tmp
export HF_HOME=~/rivermind-data/hf_home
export TORCH_HOME=~/rivermind-data/torch_home
export VLLM_CACHE_ROOT=~/rivermind-data/vllm_cache
export WANDB_DIR=~/rivermind-data/wandb
export SWANLAB_DIR=~/rivermind-data/swanlab
```

## 数据准备

仓库保留了数据处理脚本和轻量压缩包，不提交本地生成的 parquet、ChartQA 图片、日志、trace 和 checkpoint。

```bash
conda activate agent_chartqa_rt
cd ~/rivermind-data/agent_chartqa
bash scripts/prepare_local.sh
```

如果没有 ChartQA 图片，执行完整预处理：

```bash
bash data/preprocess_data.sh
```

预期生成：

- `datasets/train_full.parquet`
- `datasets/val_full.parquet`

## 训练命令

单卡 smoke / 小步数训练入口：

```bash
conda activate agent_chartqa_rt
cd ~/rivermind-data/agent_chartqa
bash scripts/train_single_gpu_smoke.sh
```

常用覆盖参数：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=~/rivermind-data/qwen/Qwen2.5-VL-3B-Instruct \
EXPERIMENT_NAME=mini_chartQA_toolfixed_growthprompt_modelonly_n4_20step_v20 \
TRAIN_MAX_STEPS=20 \
ROLLOUT_N=4 \
VERL_SAVE_MODEL_ONLY=1 \
bash scripts/train_single_gpu_smoke.sh
```

多卡入口仍保留在 `train.sh`，实际运行以命令行传入的 Hydra 参数和环境变量覆盖为准。

## 目录结构

```text
agent_chartqa/
├── agentic_eval/                 # CPU-only 训练后误差归因 harness
│   ├── agent_loop.py             # 确定性 Agent Loop（问题分类→答案抽取→约束校验→归因）
│   ├── constraints.py            # ChartQA 专用约束与错误类型分类
│   ├── memory.py                 # Lexical trace memory，支持相似失败检索
│   ├── skill_registry.py         # 可注册 skill 框架
│   ├── answer.py                 # 答案抽取与 relaxed numeric matching
│   └── cli.py                   # 命令行入口
├── data/                         # ChartQA 数据下载与 parquet 预处理
├── docs/                         # 实验报告与基线评测结果
│   ├── agentic_eval_report.md
│   ├── baseline_zeroshot_report.md
│   └── baseline_zeroshot_results.json
├── examples/
│   ├── config.yaml               # veRL/GRPO 主配置
│   ├── format_prompt/            # ChartQA prompt 模板
│   └── reward_function/          # refocus reward
├── judge/                        # LLM-judge 评测相关
├── scripts/
│   ├── prepare_local.sh
│   ├── train_single_gpu_smoke.sh
│   ├── run_baseline_eval.py      # 零样本基线评测脚本（需 GPU）
│   ├── run_baseline_eval.sh      # 一键跑基线 + v20 对比
│   ├── run_agentic_eval_smoke.sh # CPU smoke 归因测试
│   └── model_merger.py           # veRL .pt → HF safetensors 转换
├── verl/                         # veRL 训练、rollout、checkpoint 改造
├── train.sh
└── requirements.txt
```

## Agentic Evaluation Harness

本仓库已加入一个不依赖 GPU 的训练后分析模块：`agentic_eval/`。该模块参考 `medical_agent` 中的 Skill Registry、Agent Loop、记忆检索和约束校验思想，但只服务于 ChartQA trace 归因，不引入医疗知识。

核心组件：

- `agentic_eval/skill_registry.py`：把分析函数注册为 skill，并可导出 function calling schema。
- `agentic_eval/agent_loop.py`：确定性 Agent Loop，按固定顺序执行问题分类、答案抽取、约束校验、错误归因和相似失败检索。
- `agentic_eval/constraints.py`：ChartQA 专用约束，例如派生题至少需要两个 focus labels、ranking 题不能输出纯数值、必须有 `FINAL ANSWER`。
- `agentic_eval/memory.py`：纯 Python lexical trace memory，用于检索相似失败样本；后续可替换为 Milvus Lite + embedding 后端。
- `agentic_eval/cli.py`：命令行入口，可直接分析 `traces/**/trace.jsonl`。

CPU smoke：

```bash
bash scripts/run_agentic_eval_smoke.sh
```

或指定 trace 目录：

```bash
python -m agentic_eval.cli \
  --trace-path traces/mini_chartQA_toolfixed_growthprompt_modelonly_n4_20step_v20 \
  --failures-only \
  --output outputs/agentic_eval/report.jsonl
```

本地 v20 trace smoke 结果：

```json
{
  "total": 20,
  "by_error_type": {
    "correct": 14,
    "insufficient_focus_for_derived_question": 3,
    "numeric_mismatch": 2,
    "ranking_value_instead_of_category": 1
  },
  "by_question_type": {
    "absolute": 4,
    "aggregation": 2,
    "average": 2,
    "counting": 8,
    "difference": 2,
    "growth": 1,
    "ranking": 1
  }
}
```

跨版本复现实验报告已生成在：

- `docs/agentic_eval_report.md`
- `docs/agentic_eval_summary.json`

训练日志解析结果：

| Exp | Steps | Overall mean | Overall last half | Accuracy mean | Accuracy last half | Tool mean | Format mean | Throughput mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v18 | 5 | `0.9176` | `1.0000` | `0.8502` | `1.0000` | `1.0000` | `1.0000` | `205.90 tok/s` |
| v19 | 20 | `0.9163` | `0.9305` | `0.8478` | `0.8737` | `1.0000` | `1.0000` | `204.94 tok/s` |
| v20 | 20 | `0.9204` | `0.9371` | `0.8554` | `0.8857` | `1.0000` | `1.0000` | `214.85 tok/s` |

Agentic Eval 归因结果：

| Exp | Trace records | Correct | Correct rate | Failures | Main errors |
| --- | ---: | ---: | ---: | ---: | --- |
| v18 | 5 | 4 | `80.00%` | 1 | derived multi-label focus: 1 |
| v19 | 5 | 4 | `80.00%` | 1 | derived multi-label focus: 1 |
| v20 | 20 | 14 | `70.00%` | 6 | derived multi-label focus: 3, numeric mismatch: 2, ranking answer type: 1 |

这里的 trace 数量不一致，v18/v19 各保存 5 条，v20 保存 20 条。因此 trace correct rate 是诊断分布，不是直接模型优劣对比；真正可横向比较的是同样来自训练日志的 reward/accuracy/overall 曲线。

这个模块的定位是训练后自动化归因和后续 reward shaping 依据，不改变当前 v20 checkpoint 的训练结果。

## 后续可扩展方向

- Skill Registry：把 chart focus、OCR/table parsing、numeric verifier、answer normalizer 封装为可注册工具，统一转成 function calling schema。
- Agent Loop：从”单次强制工具调用”升级为”观察图表、选择工具、验证答案、必要时二次聚焦”的循环。
- Constraint Validator：限制最大工具调用次数、强制 `FINAL ANSWER` schema、禁止无证据数值、检查单位和百分号格式。
- Trace Memory/RAG：当前已实现 lexical trace memory；后续可用 Milvus Lite 存储失败 trace、题型、错误原因和修复策略，训练后分析 ranking/range/growth 类错误。
- v21 训练：将 Agentic Eval 的归因结果（ranking/growth/aggregation 聚焦不足）作为 reward shaping 依据，迭代 prompt 约束和奖励函数。
