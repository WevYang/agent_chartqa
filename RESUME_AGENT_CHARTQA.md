# Agent-ChartQA 简历描述

## 推荐写法

**Agent-ChartQA 多模态图表问答强化学习复现与工具调用优化**
`2026年05月`

- 为解决通用多模态大模型在图表问答任务中结构化定位不足、数值抽取不稳定和复杂派生推理易受语言先验影响的问题，基于 `Qwen2.5-VL-3B-Instruct`、`veRL`、`vLLM` 与 `PyTorch/FSDP` 复现 Agent-ChartQA 强化学习训练 pipeline，打通 ChartQA 数据预处理、bbox-aware 图表聚焦工具、奖励函数接入、GRPO 训练、trace 归因、model-only checkpoint 保存与 Hugging Face 发布全流程。
- 负责训练链路工程化改造：将 ChartQA 图像、问题、答案和 `x_values_bbox`/`y_values_bbox` 等结构化元数据组织为 parquet 多模态样本；重构单卡训练入口，支持模型路径、像素上限、rollout 数量、验证频率、trace 输出和缓存目录等环境变量配置，并将 Hugging Face、vLLM、W&B 等缓存统一迁移到 `~/rivermind-data`，解决系统盘空间受限问题。
- 设计并完善工具调用与奖励机制：实现鲁棒的 `FINAL ANSWER`/`ANSWER`/`\boxed{}` 解析、relaxed numeric matching 和工具执行奖励；对图表标签匹配加入大小写、单复数、星号后缀和 token-subset 模糊匹配，提升模型对 ChartQA 横/纵向柱状图标签的定位成功率；修复 vLLM rollout 进度条异常、训练 trace 图像序列化和工具执行奖励归因问题。
- 在单张 RTX 4090 48GB 环境下完成端到端 GRPO 训练验证，通过 optimizer/parameter offload、micro-batch 调整、像素裁剪和 response length 控制缓解 rollout 与 actor update 阶段显存瓶颈；20-step 复现实验中平均单步约 34s、吞吐约 215 tok/s、显存峰值约 46.6GB，并新增 model-only checkpoint 保存模式，避免完整 FSDP optimizer checkpoint 占满磁盘。
- 最终 v20 复现实验中，`reward/overall` 平均达到 0.9204、后 10 步提升至 0.9371，内部 `reward/accuracy` 平均达到 0.8553、后 10 步达到 0.8857；`reward/tool` 与 `reward/format` 全程保持 1.0，`response_clip_ratio=0`，说明模型稳定习得了先调用图表聚焦工具、再基于观察结果输出规范答案的行为；通过 trace 进一步定位 ranking、between/range 与 difference/growth 派生题仍是主要误差来源，为后续 reward shaping 和 prompt 约束优化提供依据。

## 如果要加入 Agent/RAG 技术点

建议只写成“可扩展或已设计的评测增强”，不要把医疗项目包装成 ChartQA 已完成能力。更稳妥的写法如下：

**Agent-ChartQA 多模态图表问答强化学习与 Agentic Evaluation Harness**
`2026年05月`

- 在 ChartQA 强化学习复现基础上，进一步参考多 Agent 医疗助手项目中的 Skill Registry、Agent Loop、Milvus Lite RAG、短长期记忆和 Constraint Validator 设计，规划 ChartQA 专用的 Agentic Evaluation Harness：将 chart focus、数值归一化、答案校验、错误归因和 trace 检索封装为可注册工具，服务于训练后的自动化误差分析与 reward shaping。
- 设计 trace memory 方案，将训练过程中的图表类型、工具调用参数、模型答案、标准答案、reward 分项和错误类别写入向量库，支持按 ranking、range、difference/growth 等失败模式检索相似样本，为后续 prompt 调整和奖励函数迭代提供可追溯证据。
- 设计约束校验机制，对工具调用次数、`FINAL ANSWER` 格式、单位/百分号表达、数值来源和无证据推断进行运行时校验，避免模型通过格式投机或无依据数值猜测获得局部奖励；该部分适合作为后续工程扩展，不应混同为当前 v20 checkpoint 的训练结果。

## 面试可展开点

- 为什么图表问答不只依赖文本 prompt，而需要把 `x_values_bbox`、`y_values_bbox` 等结构化视觉定位信息接入工具。
- GRPO 中 `rollout_n=4` 相比 `rollout_n=2` 为什么能提供更有效的组内奖励差异。
- 奖励函数为什么拆成 format/tool/accuracy，而不是只做最终答案 exact match。
- relaxed numeric matching 如何处理整数、小数、百分号、答案句中的多个候选数字。
- 为什么单卡 FSDP 在 world size 为 1 时会退化为 `NO_SHARD`，以及如何通过 offload、micro-batch 和像素裁剪控制显存。
- 为什么完整 optimizer checkpoint 会远大于模型权重，项目中如何用 model-only checkpoint 规避磁盘风险。
- 为什么当前结果不能写成官方 ChartQA test accuracy：v20 指标是训练内部 reward/trace 指标，官方 test accuracy 需要标准 evaluator 单独评测。
- `medical_agent` 能迁移什么：Skill Registry、Agent Loop、Milvus Lite 记忆库、约束校验、自动化评测思想。
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
