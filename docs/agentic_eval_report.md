# Agent-ChartQA Experiment Report

This report is generated from local training logs and trace JSONL files. Trace-level Agentic Eval is a smoke/diagnostic evaluation, not official ChartQA test accuracy.

## Training Metrics

| Exp | Steps | Overall mean | Overall last half | Accuracy mean | Accuracy last half | Tool mean | Format mean | Step time mean | Throughput mean | Peak reserved GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v18 | 5 | 0.9176 | 1.0000 | 0.8502 | 1.0000 | 1.0000 | 1.0000 | 34.1460 | 205.8978 | 46.6130 |
| v19 | 20 | 0.9163 | 0.9305 | 0.8478 | 0.8737 | 1.0000 | 1.0000 | 33.7598 | 204.9366 | 46.6150 |
| v20 | 20 | 0.9204 | 0.9371 | 0.8554 | 0.8857 | 1.0000 | 1.0000 | 34.1710 | 214.8470 | 46.5720 |

## Agentic Eval Trace Diagnostics

| Exp | Trace records | Correct | Correct rate | Failures | Answer score mean | Main error distribution |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v18 | 5 | 4 | 80.00% | 1 | 0.8564 | correct: 4, insufficient_focus_for_derived_question: 1 |
| v19 | 5 | 4 | 80.00% | 1 | 0.8564 | correct: 4, insufficient_focus_for_derived_question: 1 |
| v20 | 20 | 14 | 70.00% | 6 | 0.8610 | correct: 14, insufficient_focus_for_derived_question: 3, ranking_value_instead_of_category: 1, numeric_mismatch: 2 |

## Interpretation

- Training metrics show whether the policy learned the intended reward behavior: answer accuracy, tool use, format compliance, response length, throughput, and memory pressure.
- v20 improves over v19 on internal training rewards: overall mean 0.9163 -> 0.9204, overall last-half mean 0.9305 -> 0.9371, accuracy mean 0.8478 -> 0.8554, and accuracy last-half mean 0.8737 -> 0.8857.
- Tool and format rewards are saturated at 1.0 across v18/v19/v20, which means the remaining optimization target is not tool-call existence but whether the selected focus labels and final numerical/category reasoning are correct.
- Agentic Eval explains why failures happen. In v20, remaining failures are concentrated in derived multi-label focus, numeric mismatch, and ranking answer-type errors, which points to prompt/reward shaping targets.
- Trace counts are not equal across experiments: v18/v19 each have 5 saved trace records, while v20 has 20. Therefore trace correct rate is a diagnostic distribution, not a direct model-quality comparison.
- These numbers should be reported as internal reward and trace-diagnostic results. Official ChartQA accuracy requires running the standard ChartQA evaluator on the official split.

## Raw JSON

See `docs/agentic_eval_summary.json` for the machine-readable report.
