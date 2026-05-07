"""Load veRL trace JSONL files produced by Agent-ChartQA runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .schema import TraceRecord


def iter_trace_files(paths: Iterable[str | Path]) -> Iterator[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from sorted(path.glob("**/trace.jsonl"))


def load_trace_records(paths: Iterable[str | Path], limit: Optional[int] = None) -> List[TraceRecord]:
    records: List[TraceRecord] = []
    for trace_file in iter_trace_files(paths):
        with trace_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records.append(TraceRecord.from_dict(json.loads(line)))
                if limit is not None and len(records) >= limit:
                    return records
    return records
