"""Small Skill Registry adapted from the medical-agent project."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SkillParameter:
    name: str
    type: str
    description: str
    required: bool = False
    enum: Optional[List[str]] = None


class SkillRegistry:
    """Register Python callables and expose them as function-calling schemas."""

    def __init__(self) -> None:
        self._skills: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        function: Callable[..., Any],
        description: str,
        parameters: List[SkillParameter],
    ) -> None:
        self._skills[name] = {
            "function": function,
            "description": description,
            "parameters": parameters,
            "is_async": inspect.iscoroutinefunction(function),
        }

    def execute(self, name: str, **kwargs: Any) -> Any:
        skill = self._skills.get(name)
        if skill is None:
            raise KeyError(f"Skill not found: {name}")
        if skill["is_async"]:
            raise TypeError("Async skills are not supported by the CPU-only CLI loop.")
        return skill["function"](**kwargs)

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._skills)

    def to_openai_format(self) -> List[Dict[str, Any]]:
        tools = []
        for name, skill in self._skills.items():
            properties = {}
            required = []
            for param in skill["parameters"]:
                spec = {"type": param.type, "description": param.description}
                if param.enum:
                    spec["enum"] = param.enum
                properties[param.name] = spec
                if param.required:
                    required.append(param.name)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": skill["description"],
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return tools
