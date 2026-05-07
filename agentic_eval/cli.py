"""CLI for CPU-only ChartQA trace analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .agent_loop import AgenticEvaluationHarness
from .memory import LexicalTraceMemory
from .schema import TraceRecord
from .trace_loader import load_trace_records


def _demo_records() -> List[TraceRecord]:
    return [
        TraceRecord.from_dict(
            {
                "global_step": 6,
                "uid": "demo-ranking",
                "figure_id": "two_col_41319",
                "query": "What was the fifth most popular name in Belgium in 2018?",
                "ground_truth": "Alice",
                "response": "ANSWER: 422. FINAL ANSWER: 422. TERMINATE",
                "reward_overall": 0.45,
                "reward_format": 1.0,
                "reward_accuracy": 0.0,
                "reward_tool": 1.0,
                "tool_parse_status": 1,
                "tool_exec_success": 1,
                "tool_code": 'focus_on_y_values_with_draw(image_1, ["Alice"], y_values_bbox)',
            }
        ),
        TraceRecord.from_dict(
            {
                "global_step": 11,
                "uid": "demo-difference",
                "figure_id": "two_col_2289",
                "query": "What is the difference between number of mobile users in 2020 and 2024?",
                "ground_truth": "0.46",
                "response": "ANSWER: 6.95. FINAL ANSWER: 6.95. TERMINATE",
                "reward_overall": 0.4864,
                "reward_format": 1.0,
                "reward_accuracy": 0.0662,
                "reward_tool": 1.0,
                "tool_parse_status": 1,
                "tool_exec_success": 1,
                "tool_code": 'focus_on_x_values_with_draw(image_1, ["2020"], x_values_bbox)',
            }
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Agent-ChartQA trace files without GPU.")
    parser.add_argument("--trace-path", action="append", default=[], help="Trace file or directory containing trace.jsonl files.")
    parser.add_argument("--output", default="", help="Optional JSONL output path.")
    parser.add_argument("--max-records", type=int, default=0, help="Limit loaded trace records. 0 means no limit.")
    parser.add_argument("--top-k", type=int, default=3, help="Similar failure cases to attach to each analysis.")
    parser.add_argument("--accuracy-threshold", type=float, default=0.95, help="Failure threshold for trace memory indexing.")
    parser.add_argument("--failures-only", action="store_true", help="Only write non-correct analyses to output.")
    parser.add_argument("--demo", action="store_true", help="Run on built-in demo records if no local trace is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        records = _demo_records()
    else:
        if not args.trace_path:
            raise SystemExit("Provide --trace-path or use --demo.")
        records = load_trace_records(args.trace_path, limit=args.max_records or None)

    memory = LexicalTraceMemory()
    memory.build(records, accuracy_threshold=args.accuracy_threshold)
    harness = AgenticEvaluationHarness(memory=memory, top_k=args.top_k)
    analyses = harness.analyze_records(records)

    if args.failures_only:
        analyses_to_write = [item for item in analyses if item.error_type != "correct"]
    else:
        analyses_to_write = analyses

    summary = harness.summarize(analyses)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for analysis in analyses_to_write:
                handle.write(json.dumps(analysis.to_dict(), ensure_ascii=False) + "\n")
        print(f"Wrote {len(analyses_to_write)} analyses to {output_path}")


if __name__ == "__main__":
    main()
