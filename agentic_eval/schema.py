"""Shared schema for trace analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class TraceRecord:
    global_step: int = 0
    index: int = 0
    uid: str = ""
    rollout_round: int = 0
    figure_id: str = ""
    figure_path: str = ""
    query: str = ""
    ground_truth: str = ""
    response: str = ""
    penalty: float = 0.0
    reward_overall: float = 0.0
    reward_format: float = 0.0
    reward_accuracy: float = 0.0
    reward_tool: float = 0.0
    tool_parse_status: int = 0
    tool_error_code: str = ""
    tool_code: str = ""
    tool_exec_success: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceRecord":
        fields = cls.__dataclass_fields__
        kwargs = {key: data.get(key, fields[key].default) for key in fields if key != "raw"}
        kwargs["raw"] = data
        return cls(**kwargs)

    def to_search_text(self) -> str:
        return " ".join(
            [
                self.query,
                self.ground_truth,
                self.response,
                self.tool_code,
                self.figure_id,
            ]
        )


@dataclass
class ErrorAnalysis:
    uid: str
    global_step: int
    rollout_round: int
    figure_id: str
    query: str
    ground_truth: str
    prediction: str
    answer_score: float
    question_type: str
    error_type: str
    confidence: float
    violations: List[str]
    focus_labels: List[str]
    rewards: Dict[str, float]
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
