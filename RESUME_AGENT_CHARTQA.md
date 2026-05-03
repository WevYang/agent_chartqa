# Agent ChartQA 简历写法

## 一句话项目描述

复现基于 veRL 与 Qwen2.5-VL 的 ChartQA 多模态强化学习训练链路，完成数据预处理、训练脚本参数化、单卡 smoke 验证入口和本地化存储整理。

## 中文简历要点

- 复现 ChartQA 视觉问答训练项目，基于 veRL 框架与 Qwen2.5-VL-3B，梳理数据准备、reward 设计与 RL 训练配置，整理为可执行的 README 和脚本化流程。
- 编写并整理数据预处理链路，将 ChartQA 图像和 JSONL 标注转换为 parquet 多模态数据集，保留 `figure_bbox`、`x_values_bbox`、`y_values_bbox` 等结构化元数据。
- 改造训练入口，支持通过环境变量切换 GPU、模型路径、tensor parallel、实验名，并将 Hugging Face、vLLM、W&B 等缓存统一重定向到 `~/rivermind-data`，解决系统盘空间受限问题。
- 补充单卡 smoke 训练脚本，支持在 `1 x RTX 4090` 环境下快速验证训练链路，再扩展到正式多卡训练。

## 面试可展开的技术点

- 为什么 ChartQA 预处理需要把图像字节和 bbox 元数据一起写入 parquet。
- reward function 如何对 `FINAL ANSWER` 做数值相似度和多答案评分。
- 训练脚本里 `tensor_parallel_size`、`n_gpus_per_node`、batch size 之间的关系。
- 为什么要把模型下载、vLLM 缓存和实验日志重定向到数据盘。

## 英文版描述

Reproduced a ChartQA multimodal RL training pipeline based on veRL and Qwen2.5-VL, covering dataset preprocessing, parameterized training entrypoints, single-GPU smoke validation, and storage-aware runtime setup for constrained local environments.

## 英文版要点

- Reproduced a ChartQA visual question answering training project on top of veRL and Qwen2.5-VL-3B, and converted the workflow into runnable scripts and documentation.
- Built a preprocessing pipeline that converts ChartQA images and JSONL annotations into parquet-based multimodal datasets with preserved bounding-box metadata.
- Refactored the training entrypoint to support environment-driven overrides for GPU selection, model path, tensor parallelism, and experiment naming.
- Redirected Hugging Face, vLLM, and experiment caches to `~/rivermind-data` to keep training feasible on a machine with limited root-disk capacity.
