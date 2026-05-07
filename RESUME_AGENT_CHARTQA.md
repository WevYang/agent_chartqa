# Agent-ChartQA 简历描述

## 推荐写法

**Agent-ChartQA 多模态图表问答强化学习复现与工具调用优化**
`2026年05月`

- 为解决通用多模态大模型在图表问答任务中结构化定位不足、数值抽取不稳定和复杂派生推理易受语言先验影响的问题，基于 `Qwen2.5-VL-3B-Instruct`、`veRL`、`vLLM` 与 `PyTorch/FSDP` 复现 Agent-ChartQA 强化学习训练 pipeline，打通 ChartQA 数据预处理、bbox-aware 图表聚焦工具、奖励函数接入、GRPO 训练、trace 归因、model-only checkpoint 保存与 Hugging Face 发布全流程。
- 负责训练链路工程化改造：将 ChartQA 图像、问题、答案和 `x_values_bbox`/`y_values_bbox` 等结构化元数据组织为 parquet 多模态样本；重构单卡训练入口，支持模型路径、像素上限、rollout 数量、验证频率、trace 输出和缓存目录等环境变量配置，并将 Hugging Face、vLLM、W&B 等缓存统一迁移到 `~/rivermind-data`，解决系统盘空间受限问题。
- 设计并完善工具调用与奖励机制：实现鲁棒的 `FINAL ANSWER`/`ANSWER`/`\boxed{}` 解析、relaxed numeric matching 和工具执行奖励；对图表标签匹配加入大小写、单复数、星号后缀和 token-subset 模糊匹配，提升模型对 ChartQA 横/纵向柱状图标签的定位成功率；修复 vLLM rollout 进度条异常、训练 trace 图像序列化和工具执行奖励归因问题。
- 在单张 RTX 4090 48GB 环境下完成端到端 GRPO 训练验证，通过 optimizer/parameter offload、micro-batch 调整、像素裁剪和 response length 控制缓解 rollout 与 actor update 阶段显存瓶颈；20-step 复现实验中平均单步约 34s、吞吐约 215 tok/s、显存峰值约 46.6GB，并新增 model-only checkpoint 保存模式，避免完整 FSDP optimizer checkpoint 占满磁盘。
- 最终 v20 复现实验中，`reward/overall` 平均达到 0.9204、后 10 步提升至 0.9371，内部 `reward/accuracy` 平均达到 0.8553、后 10 步达到 0.8857；`reward/tool` 与 `reward/format` 全程保持 1.0，`response_clip_ratio=0`，说明模型稳定习得了先调用图表聚焦工具、再基于观察结果输出规范答案的行为；通过 trace 进一步定位 ranking、between/range 与 difference/growth 派生题仍是主要误差来源，为后续 reward shaping 和 prompt 约束优化提供依据。

## 加入 Agent/RAG 技术点后的写法

建议写成“训练后评测与误差归因增强”，不要把医疗项目包装成 ChartQA 已完成能力。更稳妥的写法如下：

**Agent-ChartQA 多模态图表问答强化学习与 Agentic Evaluation Harness**
`2026年05月`

- 在 ChartQA 强化学习复现基础上，参考多 Agent 医疗助手项目中的 Skill Registry、Agent Loop、Milvus Lite RAG、短长期记忆和 Constraint Validator 设计，落地 CPU-only 的 ChartQA Agentic Evaluation Harness：将问题类型识别、`FINAL ANSWER` 抽取、relaxed numeric verification、工具调用约束校验、错误归因和相似失败检索封装为可注册 skills，服务于训练后的自动化误差分析与 reward shaping。
- 构建 trace memory 方案，将训练过程中的图表类型、工具调用参数、模型答案、标准答案、reward 分项和错误类别写入轻量检索后端，支持按 ranking、range、difference/growth 等失败模式检索相似样本；在未安装 Milvus Lite 和 embedding 模型的环境下，先实现纯 Python lexical retrieval，后续可替换为 Milvus Lite 向量检索后端。
- 设计并实现 ChartQA 约束校验机制，对工具调用成功率、`FINAL ANSWER` 格式、ranking 题答案类型、派生题多标签聚焦、单位/百分号表达和无证据数值推断进行运行时校验；本地 v20 trace smoke 中自动归因出 14 条 correct、3 条派生题聚焦不足、2 条数值不匹配和 1 条 ranking 输出类型错误，为后续 prompt 调整和奖励函数迭代提供可追溯证据。

## 最终完整项目描述

**Agent-ChartQA 多模态图表问答强化学习复现与 Agentic 误差归因**
`2026年05月`

- 面向图表问答中结构化视觉定位不足、数值抽取不稳定和复杂派生推理易受语言先验影响的问题，基于 `Qwen2.5-VL-3B-Instruct`、`veRL`、`vLLM` 与 `PyTorch/FSDP` 复现 Agent-ChartQA 强化学习训练链路，完成 ChartQA 数据预处理、bbox-aware 图表聚焦工具、GRPO 训练、trace 归因、model-only checkpoint 保存和 Hugging Face 发布闭环。
- 负责训练链路工程化改造：将 ChartQA 图像、问题、答案与 `x_values_bbox`/`y_values_bbox` 结构化定位信息组织为 parquet 多模态样本；重构单卡训练入口，支持模型路径、像素上限、rollout 数量、验证频率、trace 输出和缓存目录配置，并将模型、日志和缓存统一迁移至 `~/rivermind-data`，解决系统盘空间受限问题。
- 设计工具调用与奖励机制：实现鲁棒 `FINAL ANSWER`/`ANSWER`/`\boxed{}` 解析、relaxed numeric matching、工具执行奖励和格式奖励；对图表标签匹配加入大小写、单复数、星号后缀和 token-subset 模糊匹配，并修复 vLLM rollout 进度条异常、训练 trace 图像序列化和工具奖励归因问题。
- 在单张 RTX 4090 48GB 上完成 20-step 端到端 GRPO 训练，通过 offload、micro-batch 调整、像素裁剪和 response length 控制缓解 rollout 与 actor update OOM；v20 实验平均单步约 34.17s、吞吐约 214.85 tok/s、显存峰值约 46.57GB，并通过 model-only checkpoint 将最终 16.26GB 权重上传至 Hugging Face，避免完整 optimizer checkpoint 占满磁盘。
- 训练指标上，v20 相比 v19 的 `reward/overall` 均值从 0.9163 提升到 0.9204、后半程从 0.9305 提升到 0.9371；`reward/accuracy` 均值从 0.8478 提升到 0.8554、后半程从 0.8737 提升到 0.8857；`reward/tool` 与 `reward/format` 在 v18/v19/v20 中均保持 1.0，说明工具调用和输出格式已稳定收敛，剩余瓶颈主要来自工具聚焦是否充分和最终数值/类别推理是否正确。
- 进一步参考多 Agent 医疗助手项目中的 Skill Registry、Agent Loop、Trace Memory/RAG 和 Constraint Validator 思想，落地 CPU-only Agentic Evaluation Harness，将题型识别、答案抽取、数值校验、工具调用约束检查、错误归因和相似失败样本检索封装为可注册 skills；在 v20 的 20 条 trace smoke 中自动归因出 14 条正确、3 条派生题聚焦不足、2 条数值不匹配和 1 条 ranking 输出类型错误，为后续 prompt 约束和 reward shaping 提供可追溯依据。

一句话解释 Agent/RAG 扩展的价值：

> 它不是直接把当前 checkpoint 分数抬高，而是把“训练完以后为什么错”变成可复现的结构化诊断：能量化错误类型、定位下一步优化方向，并避免只看平均 reward 导致误判。

## 面试可展开点

- 为什么图表问答不只依赖文本 prompt，而需要把 `x_values_bbox`、`y_values_bbox` 等结构化视觉定位信息接入工具。
- GRPO 中 `rollout_n=4` 相比 `rollout_n=2` 为什么能提供更有效的组内奖励差异。
- 奖励函数为什么拆成 format/tool/accuracy，而不是只做最终答案 exact match。
- relaxed numeric matching 如何处理整数、小数、百分号、答案句中的多个候选数字。
- 为什么单卡 FSDP 在 world size 为 1 时会退化为 `NO_SHARD`，以及如何通过 offload、micro-batch 和像素裁剪控制显存。
- 为什么完整 optimizer checkpoint 会远大于模型权重，项目中如何用 model-only checkpoint 规避磁盘风险。
- 为什么当前结果不能写成官方 ChartQA test accuracy：v20 指标是训练内部 reward/trace 指标，官方 test accuracy 需要标准 evaluator 单独评测。
- `medical_agent` 能迁移什么：Skill Registry、Agent Loop、Milvus Lite 记忆库、约束校验、自动化评测思想；当前项目已落地 CPU-only 版本，其中 Milvus Lite 作为后续可替换后端。
- `medical_agent` 不能迁移什么：医疗知识库、疾病风险判断、临床指南逻辑，这些和 ChartQA 任务无关，写进项目反而会显得不真实。

## 不建议的写法

- 不要写“在 ChartQA 官方验证集准确率达到 90%+”，因为当前没有跑官方 ChartQA evaluator。
- 不要写“融合医疗 Agent 能力提升图表问答准确率”，除非后续真的实现 ChartQA Agentic Evaluation Harness 并跑出 ablation。
- 不要写“HF 模型可直接 from_pretrained 加载”，当前上传的是 veRL/FSDP model-only state dict，不是标准 Hugging Face `save_pretrained()` 格式。

## 精简版

**Agent-ChartQA 多模态图表问答强化学习复现**
`2026年05月`

- 基于 `Qwen2.5-VL-3B-Instruct`、`veRL`、`vLLM` 与 `PyTorch/FSDP` 复现 ChartQA 图表问答强化学习链路，完成数据预处理、bbox-aware 图表聚焦工具、奖励函数、GRPO 训练、trace 归因、model-only checkpoint 保存和 Hugging Face 发布闭环。
- 改造单卡训练脚本与运行时配置，将模型、缓存、日志和数据统一迁移至 `~/rivermind-data`，通过 offload、micro-batch、像素裁剪和 response length 控制，在单张 RTX 4090 48GB 上稳定完成 20-step 端到端训练。
- 优化奖励函数与工具调用机制，实现鲁棒答案解析、relaxed numeric matching、工具执行奖励归因和图表标签模糊匹配；最终 v20 实验中 `reward/overall` 平均 0.9204、后 10 步 0.9371，`reward/accuracy` 后 10 步 0.8857，`reward/tool` 与 `reward/format` 全程 1.0，输出无截断。
