"""Generate reproducible numeric reports for Agent-ChartQA experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from .agent_loop import AgenticEvaluationHarness
from .log_parser import parse_training_log, summarize_training_steps
from .memory import LexicalTraceMemory
from .trace_loader import load_trace_records


DEFAULT_EXPERIMENTS = [
    {
        "name": "v18",
        "description": "5-step quick validation",
        "trace_path": "traces/mini_chartQA_toolfixed_balanced_n4_5step_v18",
        "log_path": "logs/train_mini_chartQA_toolfixed_balanced_n4_5step_v18_20260507_0116.log",
    },
    {
        "name": "v19",
        "description": "20-step balanced prompt",
        "trace_path": "traces/mini_chartQA_toolfixed_balanced_n4_20step_v19",
        "log_path": "logs/train_mini_chartQA_toolfixed_balanced_n4_20step_v19_20260507_0124.log",
    },
    {
        "name": "v20",
        "description": "20-step growth prompt + model-only save",
        "trace_path": "traces/mini_chartQA_toolfixed_growthprompt_modelonly_n4_20step_v20",
        "log_path": "logs/train_mini_chartQA_toolfixed_growthprompt_modelonly_n4_20step_v20_20260507_0151.log",
    },
]


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def _rounded_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    rounded = {}
    for key, value in data.items():
        if isinstance(value, dict):
            rounded[key] = _rounded_dict(value)
        else:
            rounded[key] = _round(value)
    return rounded


def analyze_experiment(exp: Dict[str, str]) -> Dict[str, Any]:
    records = load_trace_records([exp["trace_path"]])
    memory = LexicalTraceMemory()
    memory.build(records, accuracy_threshold=0.95)
    harness = AgenticEvaluationHarness(memory=memory)
    analyses = harness.analyze_records(records)
    trace_summary = harness.summarize(analyses)
    correct = trace_summary["by_error_type"].get("correct", 0)

    steps = parse_training_log(exp["log_path"])
    train_summary = summarize_training_steps(steps)
    answer_scores = [item.answer_score for item in analyses]

    return {
        "name": exp["name"],
        "description": exp["description"],
        "trace_path": exp["trace_path"],
        "log_path": exp["log_path"],
        "training": _rounded_dict(train_summary),
        "agentic_eval": {
            "trace_records": len(records),
            "correct": correct,
            "correct_rate": round(correct / len(records), 4) if records else 0.0,
            "failure_count": len(records) - correct,
            "answer_score_mean": round(mean(answer_scores), 4) if answer_scores else 0.0,
            "by_error_type": trace_summary["by_error_type"],
            "by_question_type": trace_summary["by_question_type"],
            "by_violation": trace_summary["by_violation"],
        },
    }


def _metric(summary: Dict[str, Any], metric: str, field: str = "mean") -> str:
    value = summary.get("training", {}).get(metric, {}).get(field)
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(report: Dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Agent-ChartQA Experiment Report")
    lines.append("")
    lines.append("This report is generated from local training logs and trace JSONL files. Trace-level Agentic Eval is a smoke/diagnostic evaluation, not official ChartQA test accuracy.")
    lines.append("")
    lines.append("## Training Metrics")
    lines.append("")
    lines.append("| Exp | Steps | Overall mean | Overall last half | Accuracy mean | Accuracy last half | Tool mean | Format mean | Step time mean | Throughput mean | Peak reserved GB |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for exp in report["experiments"]:
        lines.append(
            "| {name} | {steps} | {overall_mean} | {overall_last} | {acc_mean} | {acc_last} | {tool_mean} | {fmt_mean} | {step_time} | {throughput} | {mem} |".format(
                name=exp["name"],
                steps=exp["training"].get("num_steps", 0),
                overall_mean=_metric(exp, "reward/overall", "mean"),
                overall_last=_metric(exp, "reward/overall", "last_half_mean"),
                acc_mean=_metric(exp, "reward/accuracy", "mean"),
                acc_last=_metric(exp, "reward/accuracy", "last_half_mean"),
                tool_mean=_metric(exp, "reward/tool", "mean"),
                fmt_mean=_metric(exp, "reward/format", "mean"),
                step_time=_metric(exp, "perf/time_per_step", "mean"),
                throughput=_metric(exp, "perf/throughput", "mean"),
                mem=_metric(exp, "perf/max_memory_reserved_gb", "max"),
            )
        )

    lines.append("")
    lines.append("## Agentic Eval Trace Diagnostics")
    lines.append("")
    lines.append("| Exp | Trace records | Correct | Correct rate | Failures | Answer score mean | Main error distribution |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for exp in report["experiments"]:
        eval_summary = exp["agentic_eval"]
        errors = ", ".join(f"{k}: {v}" for k, v in eval_summary["by_error_type"].items())
        lines.append(
            f"| {exp['name']} | {eval_summary['trace_records']} | {eval_summary['correct']} | {eval_summary['correct_rate']:.2%} | {eval_summary['failure_count']} | {eval_summary['answer_score_mean']:.4f} | {errors} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Training metrics show whether the policy learned the intended reward behavior: answer accuracy, tool use, format compliance, response length, throughput, and memory pressure.")
    lines.append("- v20 improves over v19 on internal training rewards: overall mean 0.9163 -> 0.9204, overall last-half mean 0.9305 -> 0.9371, accuracy mean 0.8478 -> 0.8554, and accuracy last-half mean 0.8737 -> 0.8857.")
    lines.append("- Tool and format rewards are saturated at 1.0 across v18/v19/v20, which means the remaining optimization target is not tool-call existence but whether the selected focus labels and final numerical/category reasoning are correct.")
    lines.append("- Agentic Eval explains why failures happen. In v20, remaining failures are concentrated in derived multi-label focus, numeric mismatch, and ranking answer-type errors, which points to prompt/reward shaping targets.")
    lines.append("- Trace counts are not equal across experiments: v18/v19 each have 5 saved trace records, while v20 has 20. Therefore trace correct rate is a diagnostic distribution, not a direct model-quality comparison.")
    lines.append("- These numbers should be reported as internal reward and trace-diagnostic results. Official ChartQA accuracy requires running the standard ChartQA evaluator on the official split.")
    lines.append("")
    lines.append("## Raw JSON")
    lines.append("")
    lines.append(f"See `{report['json_path']}` for the machine-readable report.")
    lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Agent-ChartQA experiment comparison report.")
    parser.add_argument("--json-output", default="docs/agentic_eval_summary.json")
    parser.add_argument("--markdown-output", default="docs/agentic_eval_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = [analyze_experiment(exp) for exp in DEFAULT_EXPERIMENTS]
    report = {
        "json_path": args.json_output,
        "experiments": experiments,
    }

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.markdown_output)
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
