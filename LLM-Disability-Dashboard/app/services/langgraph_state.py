"""State definitions for LangGraph workflows."""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class LearningSessionState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    grade_level: str
    difficulty: str
    disability: str
    student_history: List[Dict[str, Any]]
    student_response: str
    student_attempt: Dict[str, Any]
    problem: Dict[str, Any]
    thought_analysis: Dict[str, Any]
    strategies: Dict[str, Any]
    tutor_session: Dict[str, Any]
    consistency_report: Dict[str, Any]
    adaptive_plan: Dict[str, Any]
    disability_analysis: Dict[str, Any]
    metadata: Dict[str, Any]
