"""Deterministic Agent Loop for post-training ChartQA trace analysis."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

from .answer import analyze_answer
from .constraints import classify_error, classify_question, extract_focus_labels, validate_record
from .memory import LexicalTraceMemory
from .schema import ErrorAnalysis, TraceRecord
from .skill_registry import SkillParameter, SkillRegistry


class AgenticEvaluationHarness:
    """CPU-only evaluation loop inspired by the medical-agent architecture.

    The loop is deterministic rather than LLM-driven: it executes registered
    analysis skills in a fixed order so results are reproducible and cheap.
    """

    def __init__(self, memory: Optional[LexicalTraceMemory] = None, top_k: int = 3) -> None:
        self.memory = memory or LexicalTraceMemory()
        self.top_k = top_k
        self.registry = SkillRegistry()
        self._register_default_skills()

    def _register_default_skills(self) -> None:
        self.registry.register(
            "classify_question",
            classify_question,
            "Classify a ChartQA question into absolute/ranking/difference/growth/etc.",
            [SkillParameter("query", "string", "ChartQA question", required=True)],
        )
        self.registry.register(
            "extract_focus_labels",
            extract_focus_labels,
            "Extract labels requested by the chart focus tool from generated tool code.",
            [SkillParameter("tool_code", "string", "Generated tool code", required=True)],
        )
        self.registry.register(
            "analyze_answer",
            analyze_answer,
            "Extract FINAL ANSWER and compute relaxed answer score.",
            [
                SkillParameter("response", "string", "Model response", required=True),
                SkillParameter("ground_truth", "string", "Ground-truth answer", required=True),
            ],
        )

    def analyze_record(self, record: TraceRecord) -> ErrorAnalysis:
        question_type = self.registry.execute("classify_question", query=record.query)
        focus_labels = self.registry.execute("extract_focus_labels", tool_code=record.tool_code)
        answer = self.registry.execute(
            "analyze_answer",
            response=record.response,
            ground_truth=record.ground_truth,
        )
        violations = validate_record(record, answer, question_type)
        error_type, confidence = classify_error(record, answer, question_type, violations)
        similar_cases = self.memory.search(
            " ".join([record.query, question_type, error_type, " ".join(violations)]),
            top_k=self.top_k,
            exclude_uid=record.uid,
        )

        return ErrorAnalysis(
            uid=record.uid,
            global_step=record.global_step,
            rollout_round=record.rollout_round,
            figure_id=record.figure_id,
            query=record.query,
            ground_truth=record.ground_truth,
            prediction=answer.extracted_answer,
            answer_score=answer.score,
            question_type=question_type,
            error_type=error_type,
            confidence=confidence,
            violations=violations,
            focus_labels=focus_labels,
            rewards={
                "overall": record.reward_overall,
                "format": record.reward_format,
                "accuracy": record.reward_accuracy,
                "tool": record.reward_tool,
            },
            similar_cases=similar_cases,
        )

    def analyze_records(self, records: Iterable[TraceRecord]) -> List[ErrorAnalysis]:
        return [self.analyze_record(record) for record in records]

    @staticmethod
    def summarize(analyses: Iterable[ErrorAnalysis]) -> Dict[str, Dict[str, int] | int]:
        analyses = list(analyses)
        return {
            "total": len(analyses),
            "by_error_type": dict(Counter(item.error_type for item in analyses)),
            "by_question_type": dict(Counter(item.question_type for item in analyses)),
            "by_violation": dict(Counter(v for item in analyses for v in item.violations)),
        }
