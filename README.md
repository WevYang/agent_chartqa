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
├── data/                         # ChartQA 数据下载与 parquet 预处理
├── examples/
│   ├── config.yaml               # veRL/GRPO 主配置
│   ├── format_prompt/            # ChartQA prompt 模板
│   └── reward_function/          # refocus reward
├── judge/                        # 评测相关代码
├── scripts/
│   ├── prepare_local.sh
│   └── train_single_gpu_smoke.sh
├── verl/                         # veRL 训练、rollout、checkpoint 改造
├── RESUME_AGENT_CHARTQA.md       # 简历写法与面试展开点
├── train.sh
└── requirements.txt
```

## 可扩展方向

`~/rivermind-data/medical_agent` 中的医疗知识本身不应直接混入 ChartQA 项目，否则主题会变得不可信；但其中的 Agent 工程能力可以合理迁移到 ChartQA：

- Skill Registry：把 chart focus、OCR/table parsing、numeric verifier、answer normalizer 封装为可注册工具，统一转成 function calling schema。
- Agent Loop：从“单次强制工具调用”升级为“观察图表、选择工具、验证答案、必要时二次聚焦”的循环。
- Constraint Validator：限制最大工具调用次数、强制 `FINAL ANSWER` schema、禁止无证据数值、检查单位和百分号格式。
- Trace Memory/RAG：用 Milvus Lite 存储失败 trace、题型、错误原因和修复策略，训练后分析 ranking/range/growth 类错误。
- Agentic Evaluation：用 evaluator agent 自动归因错误类型，辅助 reward shaping，而不是把它写成医疗问答系统。

因此，简历上合理的表述是“借鉴多 Agent/Skill Registry/RAG/约束校验思想，规划并可扩展为 ChartQA Agentic Evaluation Harness”。不建议写成“医疗 Agent 与 ChartQA 统一项目”，也不建议声称这些扩展已经参与当前 v20 训练，除非后续实际实现并跑出日志。
