"""Central registry for workflow step handlers used by the LangGraph orchestrator."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict

from app.services.adaptive_difficulty import get_adaptive_difficulty
from app.services.consistency_validator import validate_consistency
from app.services.openai_service import (
    Attempt,
    IdentifyDisability,
    Problem,
    Strategies,
    Thought,
    Tutor,
)

PromptHandler = Callable[..., Awaitable[Any]]


class PromptRegistry:
    """Registry that exposes workflow handlers by semantic name."""

    def __init__(self) -> None:
        self._registry: Dict[str, PromptHandler] = {
            "generate_problem": Problem,
            "simulate_student": Attempt,
            "analyze_thought": Thought,
            "teaching_strategies": Strategies,
            "tutor_session": Tutor,
            "identify_disability": IdentifyDisability,
            "consistency_validation": validate_consistency,
            "adaptive_difficulty": get_adaptive_difficulty,
        }

    def get(self, key: str) -> PromptHandler:
        try:
            return self._registry[key]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Prompt '{key}' is not registered") from exc

    def has(self, key: str) -> bool:
        return key in self._registry


__all__ = ["PromptRegistry", "PromptHandler"]
