# ChartQA Baseline Eval — baseline_zeroshot

- **Model**: `/root/rivermind-data/qwen_2.5_3b_instruct`
- **Val set**: `/root/rivermind-data/agent_chartqa/datasets/val_small_128.parquet` (128 samples)
- **Inference**: single-pass (no tool executor)
- **Scoring**: same `compute_score()` weights as GRPO training (accuracy×0.55 + format×0.15 + tool×0.30)

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| reward/overall mean        | 0.2760 |
| reward/overall last half   | 0.2103 |
| reward/accuracy mean       | 0.5018 |
| reward/accuracy last half  | 0.3823 |
| reward/format mean         | 0.0000 |
| reward/tool mean           | 0.0000 |

## Comparison with GRPO Training Runs

| Exp | reward/overall | reward/accuracy | reward/tool | reward/format |
|-----|---------------|----------------|------------|---------------|
| **baseline_zeroshot** (baseline) | **0.2760** | **0.5018** | **0.0000** | **0.0000** |
| v19 (GRPO 20-step) | 0.9163 | 0.8478 | 1.0000 | 1.0000 |
| v20 (GRPO growthprompt) | 0.9204 | 0.8554 | 1.0000 | 1.0000 |

> Single-pass inference does not execute the focus tool, so tool_score reflects
> whether the model *calls* the tool in its output, not whether it *uses* the result.
> GRPO-trained models additionally benefit from the multi-turn observation in the agentic loop.

## Sample Predictions (first 5)

**Q**: What is the average of net worth of the first three actors in the graph?  
**GT**: `15`  
**Pred**: ````json
[
	{"bbox_2d": [0, 0, 800, 492], "label": "image_1"}
]
````  
**Score**: overall=0.000, acc=0.000, tool=0.000

**Q**: What's the percentage of people that are at least somewhat favorable toward Brett Kavanaugh?  
**GT**: `32`  
**Pred**: `The percentage of people who are at least somewhat favorable toward Brett Kavanaugh is 32%.`  
**Score**: overall=0.550, acc=1.000, tool=0.000

**Q**: What is the ratio of fixed broadband in the years 2018 and 2019?  
**GT**: `1`  
**Pred**: `The ratio of fixed broadband in the years 2018 and 2019 is 0.994444444.`  
**Score**: overall=0.547, acc=0.994, tool=0.000

**Q**: What is the value of shortest light blue bar?  
**GT**: `2.5`  
**Pred**: `The value of the shortest light blue bar is 2.4.`  
**Score**: overall=0.528, acc=0.960, tool=0.000

**Q**: What is the difference between highest revenue of Thomas cook and lowest revenue of Kuoni  over the years ?  
**GT**: `6.6`  
**Pred**: `The difference between the highest revenue of Thomas Cook and the lowest revenue of Kuoni over the years is 6.6.`  
**Score**: overall=0.550, acc=1.000, tool=0.000

