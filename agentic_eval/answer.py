"""Answer extraction and relaxed matching utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


FINAL_ANSWER_PATTERNS = [
    re.compile(r"FINAL ANSWER:\s*(.*?)(?=\s*TERMINATE\b|\n|$)", re.DOTALL | re.IGNORECASE),
    re.compile(r"ANSWER:\s*(.*?)(?=\s*FINAL ANSWER:|\s*TERMINATE\b|\n|$)", re.DOTALL | re.IGNORECASE),
    re.compile(r"\\boxed\{([^}]*)\}", re.DOTALL),
]


@dataclass
class AnswerInfo:
    extracted_answer: str
    normalized_answer: str
    normalized_ground_truth: str
    score: float
    answer_is_numeric: bool
    ground_truth_is_numeric: bool
    numeric_candidates: List[str]


def normalize_answer(text: object) -> str:
    value = "" if text is None else str(text)
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*TERMINATE\s*$", "", value, flags=re.IGNORECASE)
    value = value.replace(",", "").replace("$", "").replace("%", "")
    return value.strip(" .,:;`'\"").lower()


def extract_final_answer(response: str) -> str:
    response = response or ""
    for pattern in FINAL_ANSWER_PATTERNS:
        match = pattern.search(response)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def is_number(text: str) -> bool:
    try:
        float(normalize_answer(text))
        return True
    except ValueError:
        return False


def numeric_candidates(text: str) -> List[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?", normalize_answer(text or ""))


def numeric_similarity(prediction: float, target: float) -> float:
    if prediction == target:
        return 1.0
    if prediction == 0 or target == 0:
        return 0.0
    return max(0.0, 1.0 - abs(prediction - target) / max(abs(prediction), abs(target)))


def relaxed_answer_score(answer: str, ground_truth: str) -> float:
    answer_norm = normalize_answer(answer)
    gt_norm = normalize_answer(ground_truth)

    if not answer_norm or not gt_norm:
        return 0.0
    if answer_norm == gt_norm:
        return 1.0

    if is_number(gt_norm):
        candidates = [answer_norm] + numeric_candidates(answer_norm)
        scores = []
        for candidate in candidates:
            if is_number(candidate):
                scores.append(numeric_similarity(float(normalize_answer(candidate)), float(gt_norm)))
        return max(scores) if scores else 0.0

    return 1.0 if answer_norm == gt_norm else 0.0


def analyze_answer(response: str, ground_truth: str) -> AnswerInfo:
    extracted = extract_final_answer(response)
    normalized = normalize_answer(extracted)
    normalized_gt = normalize_answer(ground_truth)
    candidates = numeric_candidates(extracted)
    return AnswerInfo(
        extracted_answer=extracted,
        normalized_answer=normalized,
        normalized_ground_truth=normalized_gt,
        score=relaxed_answer_score(extracted, ground_truth),
        answer_is_numeric=is_number(normalized),
        ground_truth_is_numeric=is_number(normalized_gt),
        numeric_candidates=candidates,
    )


def maybe_float(text: str) -> Optional[float]:
    if not is_number(text):
        return None
    return float(normalize_answer(text))
