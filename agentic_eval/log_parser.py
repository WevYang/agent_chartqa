"""Parse console training logs emitted by the local veRL runs."""

from __future__ import annotations

import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Keep metric indentation after the Ray actor prefix. The console log uses
# "(Runner pid=...)   key:"; consuming all whitespace would flatten sections.
PREFIX_RE = re.compile(r"^\([^)]*\)\s?")
STEP_RE = re.compile(r"^Step\s+(\d+)$")
KV_RE = re.compile(r"^([A-Za-z0-9_./-]+):\s*(.+)$")


def _clean_line(line: str) -> str:
    line = ANSI_RE.sub("", line.rstrip("\n"))
    return PREFIX_RE.sub("", line)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"nan", "inf", "-inf"}:
        return value
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_training_log(path: str | Path) -> List[Dict[str, Any]]:
    """Return one flat metric dict per logged training step."""

    steps: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    section_by_indent: Dict[int, str] = {}

    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = _clean_line(raw_line)
            if not line.strip():
                continue

            text = line.strip()
            step_match = STEP_RE.match(text)
            if step_match:
                if current is not None:
                    steps.append(current)
                current = {"step": int(step_match.group(1))}
                section_by_indent = {}
                continue

            if current is None:
                continue

            indent = len(line) - len(line.lstrip(" "))
            if text.endswith(":") and not KV_RE.match(text):
                section_by_indent[indent] = text[:-1]
                for key in list(section_by_indent):
                    if key > indent:
                        section_by_indent.pop(key, None)
                continue

            kv_match = KV_RE.match(text)
            if not kv_match:
                continue

            key, value = kv_match.groups()
            path_parts = [
                section_by_indent[idx]
                for idx in sorted(section_by_indent)
                if idx < indent
            ]
            metric_name = "/".join(path_parts + [key])
            current[metric_name] = _parse_scalar(value)

    if current is not None:
        steps.append(current)
    return steps


def _numeric_values(steps: Iterable[Dict[str, Any]], metric: str) -> List[float]:
    values = []
    for step in steps:
        value = step.get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def summarize_training_steps(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics used in README/resume reporting."""

    if not steps:
        return {"num_steps": 0}

    midpoint = max(1, len(steps) // 2)
    first = steps[:midpoint]
    last = steps[-midpoint:]
    summary: Dict[str, Any] = {
        "num_steps": len(steps),
        "first_step": steps[0].get("step"),
        "last_step": steps[-1].get("step"),
    }

    metrics = [
        "reward/overall",
        "reward/accuracy",
        "reward/tool",
        "reward/format",
        "response_length/mean",
        "response_length/clip_ratio",
        "perf/time_per_step",
        "perf/throughput",
        "perf/max_memory_reserved_gb",
        "perf/max_memory_allocated_gb",
    ]
    for metric in metrics:
        values = _numeric_values(steps, metric)
        if not values:
            continue
        first_values = _numeric_values(first, metric)
        last_values = _numeric_values(last, metric)
        summary[metric] = {
            "mean": mean(values),
            "first_half_mean": mean(first_values) if first_values else None,
            "last_half_mean": mean(last_values) if last_values else None,
            "final": values[-1],
            "min": min(values),
            "max": max(values),
        }

    return summary
