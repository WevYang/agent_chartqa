# Agent-ChartQA 简历描述

## 推荐中文写法

**Agent-ChartQA 多模态图表问答强化学习复现与工具调用优化**
`2026年05月`

- 为解决通用多模态大模型在图表问答任务中结构化定位不足、数值抽取不稳定和复杂派生推理易受语言先验影响的问题，基于 Qwen2.5-VL-3B-Instruct、veRL、vLLM 与 PyTorch/FSDP 复现 Agent-ChartQA 强化学习训练 pipeline，打通 ChartQA 数据预处理、bbox-aware 图表聚焦工具、奖励函数接入、GRPO 训练、trace 归因与 checkpoint 保存全流程。
- 负责训练链路工程化改造：将 ChartQA 图像、问题、答案和 `x_values_bbox`/`y_values_bbox` 等结构化元数据组织为 parquet 多模态样本；重构单卡训练入口，支持模型路径、像素上限、rollout 数量、验证频率、trace 输出和缓存目录等环境变量配置，并将 Hugging Face、vLLM、W&B 等缓存统一迁移到 `~/rivermind-data`，解决系统盘空间受限问题。
- 设计并完善工具调用与奖励机制：修复 vLLM rollout 进度条异常、训练 trace 图像序列化问题和工具执行奖励归因问题；实现鲁棒的 `FINAL ANSWER`/`ANSWER`/`\boxed{}` 解析、relaxed numeric matching 和工具执行奖励；对图表标签匹配加入大小写、单复数、星号后缀和 token-subset 模糊匹配，提升模型对 ChartQA 横/纵向柱状图标签的定位成功率。
- 在单张 RTX 4090 48GB 环境下完成端到端 GRPO 训练验证，通过 optimizer/parameter offload、micro-batch 调整、像素裁剪和 response length 控制缓解 rollout 与 actor update 阶段显存瓶颈；20-step 复现实验中平均单步约 34s、吞吐约 215 tok/s，显存峰值约 46.6GB，并新增 model-only checkpoint 保存模式，避免完整 FSDP optimizer checkpoint 占满磁盘。
- 最终 v20 复现实验中，`reward/overall` 平均达到 0.9204、后 10 步提升至 0.9371，内部 `reward/accuracy` 平均达到 0.8553；`reward/tool` 与 `reward/format` 全程保持 1.0，`response_clip_ratio=0`，说明模型稳定习得了先调用图表聚焦工具、再基于观察结果输出规范答案的行为；通过 trace 进一步定位了 ranking、between/range 与 difference/growth 派生题仍是主要误差来源，为后续 reward shaping 和 prompt 约束优化提供依据。

## 精简版

**Agent-ChartQA 多模态图表问答强化学习复现**
`2026年05月`

- 基于 Qwen2.5-VL-3B-Instruct、veRL、vLLM 与 PyTorch/FSDP 复现 ChartQA 图表问答强化学习训练链路，完成数据预处理、bbox-aware 图表聚焦工具、奖励函数、GRPO 训练、trace 归因和 checkpoint 保存闭环。
- 改造单卡训练脚本与运行时配置，将模型、缓存、日志和数据统一迁移至 `~/rivermind-data`，通过 offload、micro-batch、像素裁剪和 response length 控制，在单张 RTX 4090 48GB 上稳定完成 20-step 端到端训练。
- 优化奖励函数与工具调用机制，实现鲁棒答案解析、relaxed numeric matching、工具执行奖励归因和图表标签模糊匹配；最终实验中 `reward/overall` 平均 0.9204、后 10 步 0.9371，`reward/tool` 与 `reward/format` 全程 1.0，输出无截断。

## 面试可展开点

- 为什么图表问答不只依赖文本 prompt，而需要把 `x_values_bbox`、`y_values_bbox` 等结构化视觉定位信息接入工具。
- GRPO 中 `rollout_n=4` 相比 `rollout_n=2` 为什么能提供更有效的组内奖励差异。
- 为什么单卡 FSDP 在 world size 为 1 时会退化为 `NO_SHARD`，以及如何通过 offload 和裁剪控制显存。
- 为什么完整 optimizer checkpoint 会远大于模型权重，项目中如何用 model-only checkpoint 规避磁盘风险。
- 当前失败样本主要集中在 ranking、range 和 difference/growth 派生题，后续可以通过任务类型识别、双标签聚焦和更细粒度 reward shaping 继续优化。
