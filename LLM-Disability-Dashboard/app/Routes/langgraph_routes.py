"""FastAPI routes exposing LangGraph-powered workflows."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from fastapi import Body
import traceback
import logging

logger = logging.getLogger("uvicorn")

from app.services.langgraph_service import (
    run_analysis_workflow,
    run_full_workflow,
    run_learning_session,
    run_problem_workflow,
    run_workflow,
    run_improvement_graph
)


class LangGraphBaseRequest(BaseModel):
    grade_level: str = Field(default="7th", description="Student grade level")
    difficulty: str = Field(default="medium", description="Problem difficulty")
    disability: str = Field(default="Dyslexia", description="Learning disability to simulate")
    student_history: Optional[Union[List[Dict[str, Any]], str]] = None
    student_attempt: Optional[Union[Dict[str, Any], str]] = None
    student_response: Optional[str] = None
    problem: Optional[Union[Dict[str, Any], str]] = None
    metadata: Optional[Dict[str, Any]] = None
    workflow_type: Optional[str] = Field(default=None, description="Workflow variant override")

    def to_payload(self, workflow_type: Optional[str] = None) -> Dict[str, Any]:
        payload = self.model_dump(exclude_none=True)
        if workflow_type:
            payload["workflow_type"] = workflow_type
        return payload


class FullWorkflowRequest(LangGraphBaseRequest):
    workflow_type: str = Field(default="full", description="Workflow variant to execute")


class ProblemGenerationRequest(BaseModel):
    grade_level: str = Field(default="7th")
    difficulty: str = Field(default="medium")
    metadata: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        payload = self.model_dump(exclude_none=True)
        payload["workflow_type"] = "problem_only"
        return payload


class AnalysisWorkflowRequest(LangGraphBaseRequest):
    student_attempt: Union[Dict[str, Any], str]
    problem: Union[Dict[str, Any], str]
    workflow_type: str = Field(default="analysis_only")


langgraph_router = APIRouter()


@langgraph_router.get("/")
async def healthcheck() -> Dict[str, Any]:
    return {
        "status": "ok",
        "workflows": ["problem_only", "full", "analysis_only", "pre_tutor"],
    }


@langgraph_router.post("/full-workflow")
async def run_langgraph_full_workflow(payload: FullWorkflowRequest) -> Dict[str, Any]:
    try:
        print("you called me")
        return await run_full_workflow(payload.to_payload("full"))
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@langgraph_router.post("/generate-problem")
async def generate_problem(payload: ProblemGenerationRequest) -> Dict[str, Any]:
    print("--- REQUEST RECEIVED ---", flush=True)  # Force print
    try:
        print("Invoking LangGraph...", flush=True)
        return await run_problem_workflow(payload.to_payload())
    except HTTPException as e:
        print(f"CRITICAL ERROR: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    except Exception as e:  # pragma: no cover - runtime safety
        print(f"CRITICAL ERROR: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@langgraph_router.post("/analysis")
async def run_analysis(payload: AnalysisWorkflowRequest) -> Dict[str, Any]:
    try:
        return await run_analysis_workflow(payload.to_payload("analysis_only"))
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@langgraph_router.post("/workflow")
async def run_dynamic_workflow(payload: LangGraphBaseRequest) -> Dict[str, Any]:
    """Generic entrypoint that respects the payload's `workflow_type`."""

    try:
        return await run_workflow(payload.to_payload(payload.workflow_type))
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@langgraph_router.post("/session")
async def run_langgraph_session(payload: LangGraphBaseRequest) -> Dict[str, Any]:
    """Legacy endpoint kept for backwards compatibility."""

    try:
        return await run_learning_session(payload.to_payload())
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@langgraph_router.post("/improvement_analysis")
async def run_improvement_analysis(payload: dict = Body(...)):
    try:
        return await run_improvement_graph(payload)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc
