# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict, List, Optional

from mathruler.grader import extract_boxed_content, grade_answer


FINAL_ANSWER_PATTERNS = [
    re.compile(r"FINAL ANSWER:\s*(.*?)(?=\s*TERMINATE\b|\n|$)", re.DOTALL),
    re.compile(r"ANSWER:\s*(.*?)(?=\s*TERMINATE\b|\n|$)", re.DOTALL),
    re.compile(r"\\boxed\{([^}]*)\}", re.DOTALL),
]

TOOL_CALL_PATTERN = re.compile(
    r"\b(?:focus_on_columns_with_(?:mask|draw|highlight)"
    r"|focus_on_rows_with_(?:mask|draw|highlight)"
    r"|focus_on_x_values_with_(?:mask|draw|highlight)"
    r"|focus_on_y_values_with_(?:mask|draw|highlight))\s*\(",
    re.DOTALL,
)


def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def is_number(s):
    try:
        float(s)  # Will handle both int and float strings
        return True
    except ValueError:
        return False
    
def similarity_score(a, b):
    if a == b:
        return 1.0
    if a == 0 or b == 0:
        return 0.0
    return 1 - (abs(a - b) / max(abs(a), abs(b)))


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    text = text.strip(" .,:;`'\"")
    text = re.sub(r"\s*TERMINATE\s*$", "", text, flags=re.IGNORECASE).strip()
    return text.lower()


def extract_final_answer_candidates(predict: str) -> List[str]:
    candidates = []
    for pattern in FINAL_ANSWER_PATTERNS:
        match = pattern.search(predict)
        if match:
            candidate = normalize_answer(match.group(1))
            if candidate:
                candidates.append(candidate)

    if not candidates:
        tail = predict.splitlines()[-1] if predict.splitlines() else predict
        tail = normalize_answer(tail)
        if tail:
            candidates.append(tail)

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def score_single_answer(answer: str, gt: str) -> float:
    answer = normalize_answer(answer)
    gt = normalize_answer(gt)

    if is_number(gt) and is_number(answer):
        return similarity_score(float(gt), float(answer))
    return 1.0 if answer == gt else 0.0


def extract_numeric_candidates(text: str) -> List[str]:
    text = normalize_answer(text)
    # ChartQA commonly uses relaxed numeric matching. If the model writes a
    # sentence in ANSWER without FINAL ANSWER, score the numeric values inside it.
    return re.findall(r"[-+]?\d+(?:\.\d+)?", text)


def tool_use_score(predict: str, penalty: Optional[float] = None) -> float:
    text_score = 1.0 if TOOL_CALL_PATTERN.search(predict) else 0.0
    if penalty is None:
        return text_score
    if penalty > 0:
        return max(text_score, 1.0)
    if penalty < 0:
        return 0.0
    return text_score


def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    queries: Optional[List[str]] = None,
    penalties: Optional[List[float]] = None,
    format_weight: float = 0.15,
    tool_weight: float = 0.30,
) -> List[Dict[str, float]]:
    scores = []
    for idx, (predict, ground_truth) in enumerate(zip(predicts, ground_truths)):
        candidates = extract_final_answer_candidates(predict)
        sub_gts = [gt for gt in ground_truth.split("|||") if gt.strip()]
        if not sub_gts:
            sub_gts = [ground_truth]

        format_score = 1.0 if any(
            pattern.search(predict) for pattern in FINAL_ANSWER_PATTERNS[:-1]
        ) else 0.0

        best_accuracy = 0.0
        for candidate in candidates:
            candidate_answers = [part for part in candidate.split("||") if part.strip()]
            if not candidate_answers:
                continue

            correct_answers = 0.0
            for answer in candidate_answers:
                expanded_answers = [answer]
                if any(is_number(normalize_answer(gt)) for gt in sub_gts):
                    expanded_answers.extend(extract_numeric_candidates(answer))
                candidate_scores = [
                    score_single_answer(expanded_answer, gt)
                    for expanded_answer in expanded_answers
                    for gt in sub_gts
                ]
                if candidate_scores:
                    correct_answers += max(candidate_scores)

            expected_answers = max(len(sub_gts), 1)
            best_accuracy = max(best_accuracy, correct_answers / expected_answers)

        penalty = penalties[idx] if penalties is not None and idx < len(penalties) else None
        tool_score = tool_use_score(predict, penalty=penalty)
        accuracy_weight = max(0.0, 1.0 - format_weight - tool_weight)
        overall_score = (
            accuracy_weight * best_accuracy
            + format_weight * format_score
            + tool_weight * tool_score
        )
        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "accuracy": best_accuracy,
                "tool": tool_score,
            }
        )

    return scores
