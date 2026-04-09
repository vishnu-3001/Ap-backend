# AP Backend — LLM Disability Dashboard (FastAPI)

Backend service for an **AI-powered educational dashboard** that generates math problems, simulates student thinking patterns for different learning disabilities, and provides tutoring-style guidance. Built with **FastAPI + OpenAI + LangChain/LangGraph**.

---

## Why this project

This backend demonstrates how to design a production-style API around LLM workflows:

- **FastAPI REST backend** with versioned routes and CORS enabled.
- **LLM-powered content generation** (math problems, analysis, strategies, tutoring responses).
- **Workflow orchestration with LangGraph** for structured, multi-step reasoning pipelines (problem generation → attempt/analysis → tutoring).
- **Adaptive learning features** like difficulty adjustment and response consistency checks.
- Clean separation of concerns via `Routes/` and `services/`.

---

## What we built

### 1) FastAPI application + versioned APIs
The app is configured as **Educational Dashboard API** and exposes:
- `v1` routes focused on direct OpenAI service calls
- `v1/v2` routes exposing LangGraph-powered workflows

The entrypoint registers routers:
- `/api/v1/openai` (OpenAI endpoints)
- `/api/v1/langgraph` and `/api/v2/langgraph` (LangGraph workflows)

---

### 2) OpenAI-powered endpoints (v1)

Base path: `/api/v1/openai`

Key capabilities:
- **Problem generation**
  - `GET /generate_problem?grade_level=7th&difficulty=medium`
- **Thought / reasoning simulation** (conditioned on disability + optional student attempt)
  - `POST /generate_thought`
- **Strategy generation**
  - `POST /generate_strategies`
- **Generate a student attempt**
  - `POST /generate_attempt`
- **Tutor response generation**
  - `POST /generate_tutor`
- **Identify likely disability from student response**
  - `POST /identify_disability`
- **Consistency validation** between expected answer and student attempt
  - `POST /validate_consistency`
- **Adaptive difficulty** based on student history
  - `POST /adaptive_difficulty`
- **General chat endpoint** (tutor-style / configurable personality)
  - `POST /chat`

This shows practical API design around LLM features: structured payloads, safe defaults, and clear error handling.

---

### 3) LangGraph-powered workflows (v1/v2)

Base path: `/api/v1/langgraph` (also mounted at `/api/v2/langgraph`)

Included workflows:
- `GET /` healthcheck (returns supported workflow types)
- `POST /generate-problem` (LangGraph problem-only flow)
- `POST /full-workflow` (end-to-end orchestration)
- `POST /analysis` (analysis-only path)
- `POST /workflow` (dynamic workflow selector via `workflow_type`)
- `POST /session` (legacy session endpoint)
- `POST /improvement_analysis` (improvement-focused analysis graph)

This highlights the ability to move beyond “single prompt calls” into **repeatable, testable multi-step reasoning graphs**.

---

## Tech stack

- **Python**
- **FastAPI** + **Uvicorn/Gunicorn**
- **OpenAI SDK**
- **LangChain** (`langchain`, `langchain-openai`, etc.)
- **LangGraph** (`langgraph`, checkpoints, prebuilt)

See `requirements.txt` for pinned versions (LangChain 0.3.x + LangGraph 0.5.x).

---

## Running locally

### 1) Setup
```bash
cd LLM-Disability-Dashboard
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Environment variables
Create a `.env` file (or export env vars) with your API keys (example):
```bash
OPENAI_API_KEY=your_key_here
```

### 3) Start the API
```bash
python main.py
```

Server will run on:
- `http://localhost:8000`

Swagger UI:
- `http://localhost:8000/docs`

---

## Example requests

### Generate a problem (OpenAI route)
```bash
curl "http://localhost:8000/api/v1/openai/generate_problem?grade_level=7th&difficulty=medium"
```

### Run LangGraph healthcheck
```bash
curl "http://localhost:8000/api/v1/langgraph/"
```

---

## Project structure (high level)

- `LLM-Disability-Dashboard/main.py` — FastAPI app entrypoint + router mounting
- `LLM-Disability-Dashboard/app/Routes/` — API route definitions
  - `openai_routes.py` — OpenAI feature endpoints
  - `langgraph_routes.py` — LangGraph workflow endpoints
- `LLM-Disability-Dashboard/app/services/` — business logic (LLM calls, workflows, validators, adaptive difficulty)

---

## What this demonstrates to recruiters

- Building a real backend around LLM features (not just notebooks)
- API design: versioning, structured payloads, error handling, extensibility
- Orchestrating multi-step LLM workflows using LangGraph
- Applied “AI for education” domain thinking: tutoring, disability-aware reasoning, adaptive difficulty

---

## Notes / Next improvements (optional roadmap)

- Restrict CORS origins for production
- Add authentication / rate limiting
- Add automated tests for route contracts and workflow outputs
- Add Dockerfile + deployment docs
