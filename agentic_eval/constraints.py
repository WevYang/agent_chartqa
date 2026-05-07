"""ChartQA-specific constraints and error taxonomy."""

from __future__ import annotations

import ast
import re
from typing import List

from .answer import AnswerInfo, extract_final_answer
from .schema import TraceRecord


QUESTION_PATTERNS = [
    ("difference", re.compile(r"\b(difference|gap|how much more|how much less|between)\b", re.I)),
    ("growth", re.compile(r"\b(growth|increase|decrease|change|rose|fell)\b", re.I)),
    ("ranking", re.compile(r"\b(largest|smallest|highest|lowest|most|least|rank|fifth|fourth|third|second)\b", re.I)),
    ("aggregation", re.compile(r"\b(sum|total|combined|altogether)\b", re.I)),
    ("average", re.compile(r"\b(average|mean)\b", re.I)),
    ("counting", re.compile(r"\b(how many|number of)\b", re.I)),
]


def classify_question(query: str) -> str:
    for label, pattern in QUESTION_PATTERNS:
        if pattern.search(query or ""):
            return label
    return "absolute"


def extract_focus_labels(tool_code: str) -> List[str]:
    labels: List[str] = []
    for match in re.finditer(r"\[[^\]]*\]", tool_code or ""):
        try:
            parsed = ast.literal_eval(match.group(0))
        except Exception:
            continue
        if isinstance(parsed, list):
            labels.extend(str(item) for item in parsed)

    deduped = []
    seen = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            deduped.append(label)
    return deduped


def validate_record(record: TraceRecord, answer: AnswerInfo, question_type: str) -> List[str]:
    violations: List[str] = []

    if not extract_final_answer(record.response):
        violations.append("missing_final_answer")
    if record.reward_tool < 1.0:
        violations.append("missing_or_unrewarded_tool_call")
    if int(record.tool_parse_status or 0) != 1:
        violations.append("tool_parse_failed")
    if int(record.tool_exec_success or 0) != 1:
        violations.append("tool_execution_failed")
    if record.reward_format < 1.0:
        violations.append("format_not_satisfied")

    focus_labels = extract_focus_labels(record.tool_code)
    # sum/aggregation questions also need at least 2 focus labels to compute correctly
    if question_type in {"difference", "growth", "aggregation"} and len(focus_labels) < 2:
        violations.append("derived_question_needs_two_or_more_focus_labels")

    if question_type == "ranking" and answer.answer_is_numeric and not answer.ground_truth_is_numeric:
        # Distinguish: did the model focus on the correct category label?
        # If GT appears in focus_labels, the model identified the right category
        # but read its numeric bar value instead of the label name.
        gt_norm = answer.normalized_ground_truth
        gt_in_focus = any(
            gt_norm in fl.lower() or fl.lower() in gt_norm
            for fl in focus_labels
            if fl.strip()
        )
        if gt_in_focus:
            violations.append("correct_focus_but_read_numeric_value")
        else:
            violations.append("ranking_answer_should_be_category_not_number")

    if answer.ground_truth_is_numeric and not answer.numeric_candidates and answer.normalized_answer:
        violations.append("numeric_ground_truth_but_no_numeric_prediction")

    return violations


# Answers within this score are considered correct (relaxed numeric matching).
CORRECT_SCORE_THRESHOLD = 0.90


def classify_error(record: TraceRecord, answer: AnswerInfo, question_type: str, violations: List[str]) -> tuple[str, float]:
    if answer.score >= CORRECT_SCORE_THRESHOLD or record.reward_accuracy >= CORRECT_SCORE_THRESHOLD:
        return "correct", 0.99
    if "tool_parse_failed" in violations or "tool_execution_failed" in violations:
        return "tool_failure", 0.95
    if "missing_final_answer" in violations or "format_not_satisfied" in violations:
        return "format_failure", 0.90
    if "correct_focus_but_read_numeric_value" in violations:
        # Model identified the right category but reported its bar value instead of the label name.
        return "category_reading_error", 0.90
    if "ranking_answer_should_be_category_not_number" in violations:
        return "ranking_value_instead_of_category", 0.95
    if "derived_question_needs_two_or_more_focus_labels" in violations:
        return "insufficient_focus_for_derived_question", 0.92
    if question_type in {"difference", "growth"} and answer.ground_truth_is_numeric:
        return "derived_numeric_reasoning_error", 0.82
    if question_type == "ranking":
        return "ranking_selection_error", 0.80
    if answer.ground_truth_is_numeric:
        return "numeric_mismatch", 0.76
    return "semantic_mismatch", 0.70
