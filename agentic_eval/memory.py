"""Trace memory backends for retrieval over failed cases."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from .schema import ErrorAnalysis, TraceRecord


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[token] * b.get(token, 0) for token in a)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LexicalTraceMemory:
    """Dependency-free retrieval backend for trace failures.

    This is intentionally lightweight so the harness can run before GPU and
    before Milvus/SentenceTransformer are installed. The interface can be
    replaced with a Milvus Lite vector backend later.
    """

    def __init__(self) -> None:
        self._items: List[Dict[str, Any]] = []

    def add(self, record: TraceRecord, analysis: Optional[ErrorAnalysis] = None) -> None:
        text = record.to_search_text()
        if analysis is not None:
            text = f"{text} {analysis.question_type} {analysis.error_type} {' '.join(analysis.violations)}"
        self._items.append(
            {
                "uid": record.uid,
                "global_step": record.global_step,
                "figure_id": record.figure_id,
                "query": record.query,
                "ground_truth": record.ground_truth,
                "reward_accuracy": record.reward_accuracy,
                "text": text,
                "vector": Counter(_tokenize(text)),
            }
        )

    def build(self, records: Iterable[TraceRecord], accuracy_threshold: float = 0.95) -> None:
        for record in records:
            if record.reward_accuracy < accuracy_threshold:
                self.add(record)

    def search(self, query: str, top_k: int = 3, exclude_uid: str = "") -> List[Dict[str, Any]]:
        query_vector = Counter(_tokenize(query))
        scored = []
        for item in self._items:
            if exclude_uid and item["uid"] == exclude_uid:
                continue
            score = _cosine(query_vector, item["vector"])
            if score <= 0:
                continue
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        results = []
        for score, item in scored[:top_k]:
            results.append(
                {
                    "score": round(score, 4),
                    "uid": item["uid"],
                    "global_step": item["global_step"],
                    "figure_id": item["figure_id"],
                    "query": item["query"],
                    "ground_truth": item["ground_truth"],
                    "reward_accuracy": item["reward_accuracy"],
                }
            )
        return results
