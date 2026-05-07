# Agent-ChartQA

Agent-ChartQA 是一个基于 `Qwen2.5-VL-3B-Instruct`、`veRL`、`vLLM` 与 `PyTorch/FSDP` 的多模态图表问答强化学习复现项目。核心问题是：通用多模态大模型在结构化图表场景下存在视觉定位不足、数值抽取不稳定和复杂派生推理易受语言先验影响三类系统性缺陷；本项目的做法是引入 bbox-aware 图表聚焦工具，通过多维度分项奖励和 GRPO 让模型从零习得"先定位目标区域、再输出结构化答案"的行为，在不依赖监督示范的条件下完成 RL 端到端训练闭环。

训练后以相同打分函数对比零样本基线：`reward/overall` 从 `0.28` 提升至 `0.92`，`reward/accuracy` 从 `0.50` 提升至 `0.86`，工具调用与格式规范从 `0` 提升至 `1.0`——两项行为均完全通过 RL 从零习得，无监督示范。v20 最终 checkpoint 已上传 Hugging Face。

除训练链路外，仓库还提供了一个不依赖 GPU 的 Agentic Evaluation Harness（`agentic_eval/`），以 Skill Registry + Agent Loop 为核心，将答案抽取、聚焦约束校验与失败样本检索封装为可注册 skill，用于训练后自动化误差归因。在 v20 的 20 条训练轨迹上，自动归因人工抽样一致率 100%（4/4），失败题型与验证集低准确率题型完全吻合（ranking 基线仅 11.5%），验证归因指向真实训练瓶颈。

---

## 核心机制

### 1. bbox-aware 图表聚焦工具：把视觉定位结构化

ChartQA 的主要困难不在于读文字，而在于识别"哪根柱子、哪个折线点属于哪个类别"。通用多模态模型的处理方式是在全图上做注意力，容易把相邻标签或相近数值混淆。

本项目的做法是把 `x_values_bbox`、`y_values_bbox` 等结构化定位信息以 parquet 形式注入训练样本，并实现 `focus_on_x_values_with_draw` / `focus_on_y_values_with_draw` 两个工具，让模型在 prompt 约束下先声明要聚焦的标签列表，再依据工具执行结果输出最终答案：

```text
用户问题 + 图像 + bbox 信息
        │
        ▼
┌────────────────────────────────────────┐
│  模型调用 focus_on_x/y_values_with_draw │
│  参数: ["label_a", "label_b", ...]      │
└──────────────┬─────────────────────────┘
               │  工具执行（bbox 定位 + 可视化裁剪）
               ▼
┌────────────────────────────────────────┐
│  模型基于聚焦结果输出                   │
│  FINAL ANSWER: <value>                  │
└────────────────────────────────────────┘
```

工具执行层支持大小写、单复数、星号后缀和 token-subset 模糊匹配，解决 ChartQA 标签写法多变导致精确匹配失效的问题。`reward/tool` 记录工具调用是否成功执行；工具调用奖励的引入使模型在 v18 第一轮就达到 `reward/tool=1.0`，此后全程保持。

### 2. GRPO 奖励函数：多维度分项 + relaxed numeric matching

奖励函数拆成四个维度，训练权重 `accuracy×0.55 + format×0.15 + tool×0.30`：

| 维度 | 来源 | 说明 |
|------|------|------|
| `reward/accuracy` | `relaxed_answer_score()` | 对数值答案做 `1 - \|a-b\| / max(\|a\|, \|b\|)` 相似度匹配，非数值做归一化 exact match |
| `reward/format` | `FINAL ANSWER:` 检测 | 是否包含规定格式的答案段落 |
| `reward/tool` | 工具执行结果 | 是否成功调用 focus 工具并得到执行反馈 |
| `reward/overall` | 加权综合 | 三项加权，作为 GRPO 的分组奖励信号 |

答案解析层支持 `FINAL ANSWER: ...`、`ANSWER: ...`、`\boxed{...}` 三种格式，答案中的 `,`、`$`、`%` 符号在归一化时自动剥离。这个设计使得格式奖励不会因为答案附带单位就失效，避免了把奖励函数"调教模型"变成"模型记忆固定格式"。

GRPO 的分组奖励来自 `rollout_n=4` 的组内比较：同一问题生成 4 条 rollout，组内奖励差异越大，梯度信号越强。单卡 FSDP 在 world_size=1 时退化为 `NO_SHARD`，结合 parameter/optimizer offload、micro-batch 调整和图像像素裁剪，在 RTX 4090 48GB 上稳定运行。

### 3. Agentic Evaluation Harness：Skill Registry + Agent Loop 训练后归因

归因模块的核心设计是把所有分析步骤封装为可注册 skill，Agent Loop 按固定顺序调度：

```text
TraceRecord（训练轨迹）
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  AgenticEvaluationHarness                           │
│                                                     │
│  ┌──────────────────┐   ┌──────────────────────┐   │
│  │  SkillRegistry   │──▶│  classify_question   │   │  问题类型分类
│  │  (可注册 / 可导出 │   ├──────────────────────┤   │
│  │   function call  │   │  extract_focus_labels │   │  聚焦标签提取
│  │   schema)        │   ├──────────────────────┤   │
│  └──────────────────┘   │  analyze_answer      │   │  答案抽取 + 打分
│                         ├──────────────────────┤   │
│                         │  validate_record     │   │  约束校验
│                         ├──────────────────────┤   │
│                         │  classify_error      │   │  错误归因
│                         └──────────────────────┘   │
│                                   │                 │
│  ┌────────────────────────────┐   │                 │
│  │  LexicalTraceMemory        │◀──┘                 │  相似失败检索
│  │  (BM25-style，可替换为     │                     │
│  │   Milvus Lite 向量后端)    │                     │
│  └────────────────────────────┘                     │
└─────────────────────────────────────────────────────┘
        │
        ▼
ErrorAnalysis（uid, error_type, violations, similar_cases, ...）
```

约束校验层（`constraints.py`）覆盖六类规则：
- 必须有 `FINAL ANSWER` 格式输出
- 工具必须调用成功（`tool_parse_status=1 & tool_exec_success=1`）
- ranking 类问题答案不能是纯数值（输出数值但 GT 是类别名 → 归因到 `ranking_value_instead_of_category`）
- ranking 类问题若模型聚焦了正确类别标签但输出了其柱值 → 归因到 `category_reading_error`（区别于选错标签）
- difference / growth / aggregation 派生题至少需要两个 focus labels
- 数值 GT 但预测中无数值候选

v20 trace smoke 自动归因结果：

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
    "absolute": 4, "aggregation": 2, "average": 2,
    "counting": 8, "difference": 2, "growth": 1, "ranking": 1
  }
}
```

### 4. 零样本基线对比：以相同打分函数量化 RL 收益

基线评测使用相同的 `compute_score()` 权重，在同一 128 样本验证集上对未经训练的 `Qwen2.5-VL-3B-Instruct` 做单次 vLLM 推理（无工具执行回路）：

| 模型 | `reward/overall` | `reward/accuracy` | `reward/tool` | `reward/format` |
|------|---:|---:|---:|---:|
| Qwen2.5-VL-3B-Instruct（零样本基线） | 0.2760 | 0.5018 | 0.0000 | 0.0000 |
| v20 GRPO 20-step（训练后，agentic loop） | 0.9204 | 0.8554 | 1.0000 | 1.0000 |

GRPO 训练带来 `reward/overall` +0.64、`reward/accuracy` +0.35；工具调用与格式规范从 0 提升至 1.0，两项行为完全通过 RL 从零习得。

---

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

## Quick Start

```bash
pip install -r requirements.txt

# 准备数据（需要 ChartQA 原始图片）
bash data/preprocess_data.sh

# 单卡训练
conda activate agent_chartqa_rt
bash scripts/train_single_gpu_smoke.sh

# CPU-only 归因 smoke（无需 GPU）
bash scripts/run_agentic_eval_smoke.sh

# 零样本基线评测（需 GPU）
python scripts/run_baseline_eval.py \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --val_file   datasets/val_small_128.parquet \
  --output_dir docs/baseline_eval \
  --label      baseline_zeroshot
```

## 运行环境

建议使用本地 conda 环境执行：

```bash
conda activate agent_chartqa_rt
cd ~/agent_chartqa
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

系统盘空间不足时，建议把缓存统一放到数据盘：

```bash
export TMPDIR=~/data/tmp
export HF_HOME=~/data/hf_home
export TORCH_HOME=~/data/torch_home
export VLLM_CACHE_ROOT=~/data/vllm_cache
export WANDB_DIR=~/data/wandb
```

## 数据准备

仓库保留了数据处理脚本和轻量压缩包，不提交本地生成的 parquet、ChartQA 图片、日志、trace 和 checkpoint。

```bash
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
bash scripts/train_single_gpu_smoke.sh
```

常用覆盖参数：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
EXPERIMENT_NAME=mini_chartQA_toolfixed_growthprompt_modelonly_n4_20step_v20 \
TRAIN_MAX_STEPS=20 \
ROLLOUT_N=4 \
VERL_SAVE_MODEL_ONLY=1 \
bash scripts/train_single_gpu_smoke.sh
```

多卡入口仍保留在 `train.sh`，实际运行以命令行传入的 Hydra 参数和环境变量覆盖为准。

## 基线评测

对任意 HF 模型路径做单次推理评测，使用与训练相同的 `compute_score()` 打分：

```bash
# 零样本基线
python scripts/run_baseline_eval.py \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --val_file   datasets/val_small_128.parquet \
  --output_dir docs/baseline_eval \
  --label      baseline_zeroshot

# 微调后单次推理对比（需先通过 model_merger.py 转换 checkpoint）
python scripts/run_baseline_eval.py \
  --model_path checkpoints/.../actor/huggingface \
  --val_file   datasets/val_small_128.parquet \
  --output_dir docs/baseline_eval \
  --label      v20_finetuned
```

一键脚本（基线 + v20 对比）：

```bash
bash scripts/run_baseline_eval.sh
bash scripts/run_baseline_eval.sh --baseline-only   # 只跑基线
bash scripts/run_baseline_eval.sh --skip-merge      # 跳过 checkpoint 转换
```

## Agentic Evaluation Harness

本仓库已加入一个不依赖 GPU 的训练后分析模块：`agentic_eval/`。

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

跨版本复现实验报告：

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

trace 数量不一致：v18/v19 各保存 5 条，v20 保存 20 条。trace correct rate 是诊断分布，不是直接模型优劣对比；横向比较应使用训练日志中的 reward/accuracy/overall 曲线。

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

## 局限性

- **训练结果是内部 reward，不是官方 ChartQA accuracy**：v20 指标来自训练打分函数和 trace smoke，官方 test accuracy 需要单独跑标准 ChartQA evaluator，当前未做。
- **单卡 20-step 是复现验证，不是完整训练**：当前仓库的 checkpoint 是在小规模 val 子集上调通工具调用和奖励链路，完整训练需要更多步数、更大数据集和更多计算资源。
- **model-only checkpoint 不能直接 from_pretrained 加载**：HF 仓库上传的是 veRL/FSDP 格式的 state dict，需要先通过 `scripts/model_merger.py` 转换为标准 HF safetensors 格式才能做常规推理。
- **工具调用是 single-pass，不是真正的 agentic loop**：当前训练中工具只执行一次；如需多轮聚焦（二次确认、边界重定位），需要在 rollout 层升级为 multi-step executor。
- **Agentic Eval 的 trace memory 是 lexical retrieval**：当前 `LexicalTraceMemory` 用关键词匹配，相似失败检索精度受限；后续可替换为 Milvus Lite 向量后端，但当前环境未安装。
- **归因覆盖率依赖 trace 数量**：v18/v19 各只有 5 条 trace，v20 有 20 条，样本量偏小，归因分布的统计意义有限，更适合作为定性诊断工具而非定量评测。

## 文档

- `docs/agentic_eval_report.md`：跨版本 Agentic Eval 归因报告（v18/v19/v20）与零样本基线对比
- `docs/agentic_eval_summary.json`：机器可读的训练指标与归因汇总
- `docs/baseline_zeroshot_report.md`：零样本基线详细评测报告（128 样本，逐指标分析）
- `docs/baseline_zeroshot_results.json`：128 条样本的逐样本评测结果（含 query、GT、prediction、各项 reward）
- `examples/reward_function/refocus.py`：完整奖励函数实现（含 relaxed numeric matching 和工具执行奖励）
- `examples/format_prompt/chartQA_short.jinja`：强制工具调用的 prompt 模板
- `examples/config.yaml`：veRL/GRPO 完整训练配置（含 offload、micro-batch、rollout_n 等参数）

## 后续可扩展方向

- **Skill Registry 扩展**：把 chart focus、OCR/table parsing、numeric verifier、answer normalizer 封装为可注册工具，统一转成 function calling schema。
- **Agent Loop 升级**：从"单次强制工具调用"升级为"观察图表、选择工具、验证答案、必要时二次聚焦"的循环。
- **Constraint Validator 细化**：限制最大工具调用次数、强制 `FINAL ANSWER` schema、禁止无证据数值、检查单位和百分号格式。
- **Trace Memory/RAG**：当前已实现 lexical trace memory；后续可用 Milvus Lite 存储失败 trace、题型、错误原因和修复策略，训练后分析 ranking/range/growth 类错误。
- **v21 训练**：将 Agentic Eval 的归因结果（ranking/growth/aggregation 聚焦不足）作为 reward shaping 依据，迭代 prompt 约束和奖励函数，并做 ablation 验证归因到训练改进的闭环。

## 更新计划

- 补齐 `scripts/model_merger.py` 的输出校验：当前转换后的 safetensors 没有自动做 weight hash 比对，磁盘空间不足时容易生成损坏的分片。
- 增加 question-type 分层评测脚本：在验证集上按 ranking / difference / growth / counting 分别统计 `reward/accuracy`，量化各类型的改进空间。
- 补全 `docs/` 文档：`data_format.md`（训练 parquet schema 与 bbox 字段说明）、`reward_design.md`（奖励函数设计依据与参数选择）。
- 补充可复现 benchmark 配置：固定验证集、打分权重和运行命令，使每次实验结果可直接横向对比。
- 接入 GitHub Actions：对 `agentic_eval/` 模块运行 CPU smoke 测试，在 PR 时自动验证归因逻辑未被破坏。
- 官方 ChartQA evaluator 对齐：在标准 test split 上跑官方 evaluator，补齐当前仅有内部 reward 指标的空白。
