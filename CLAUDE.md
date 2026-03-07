# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Hackathon Project: [Your Project Name]


## Project Structure

- `/frontend` — Next.js app
- `/backend` — FastAPI + LangGraph agents
- `/shared` — Shared types/schemas

## Development Setup

The `.gitignore` is configured for Python with support for common tooling: `pip`/`uv`/`poetry`/`pdm`/`pipenv`, Django, Flask, pytest, mypy, ruff, and Jupyter notebooks. Set up a virtual environment before installing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # once a requirements file exists
```

## Commands

Commands will be established as the project develops. Common patterns to follow once files are present:

- **Lint:** `ruff check .` (ruff is configured in `.gitignore`)
- **Type check:** `mypy .`
- **Tests:** `pytest` (run a single test with `pytest path/to/test_file.py::test_name`)
- **Frontend dev**: `cd frontend && npm run dev`
- **Backend dev**: `cd backend && uvicorn main:app --reload`
- **Tests**: `cd backend && pytest`

## Tech Stack

- Frontend: Next.js 15 (App Router) + TypeScript + Tailwind CSS
- Backend: Python 3.12 + FastAPI
- AI/Agents: LangGraph + LangChain
- Vector DB: Pinecone / ChromaDB (for RAG)
- Database: Supabase (PostgreSQL)
- Deployment: Vercel (frontend) + Railway (backend)

## Rules

- Use TypeScript strict mode in frontend
- Use Pydantic v2 models for all API schemas
- All agents must have clear input/output schemas
- Write tests for all agent nodes
- No console.logs in production code
- Commit messages: conventional commits format

npm install framer-motion
```

And Tailwind itself isn't bland — it's the utility layer underneath. Magic UI just gives you pre-built animated components (hero sections, cards with shimmer effects, animated borders, etc.) so you don't have to build those from scratch during a hackathon. You'll still be writing Tailwind classes for layout and spacing.

One more tip: when you're prompting Claude to build UI, reference Magic UI explicitly so it pulls from the MCP instead of defaulting to plain Tailwind:
```
"Build the landing page using Magic UI components for the hero 
section and feature cards. Use animated-beam for the agent 
visualization and shimmer-button for the CTA."


# CyberCypher 5.0 — Agentic AI for Autonomous Network Operations

## What We're Building
An agentic AI system that acts as an autonomous network operations 
layer for an ISP. Core loop: Observe → Reason → Decide → Act → Learn.

## Tech Stack
- Backend: Python 3.12 + FastAPI
- Agent Framework: LangGraph (StateGraph with conditional edges)
- LLM: Claude API via LangChain
- Simulated Telemetry: Python generators producing realistic network signals
- Vector DB: ChromaDB (for RAG over network runbooks/documentation)
- Frontend: Next.js + TypeScript + Tailwind + Magic UI
- Database: SQLite (lightweight, no infra needed for hackathon)

## Architecture
- /backend — FastAPI + LangGraph agent system
  - /agents — Individual agent nodes (observer, reasoner, decider, actor, learner)
  - /simulation — Network telemetry simulator
  - /models — Pydantic schemas for all data
  - /tools — Agent tools (reroute traffic, rate limit, rollback, escalate)
- /frontend — Dashboard showing agent reasoning and network state
- /data — Sample telemetry data, runbooks for RAG

## Key Constraints
- Every agent action must be logged with reasoning (explainability)
- Define clear autonomy boundaries: auto-act vs human-approval-required
- Agent must handle partial observability (incomplete/noisy data)
- All agent nodes must have typed input/output schemas
- Safety guardrails: blast radius checks before any action

## Commands
- Backend: cd backend && uvicorn main:app --reload
- Frontend: cd frontend && npm run dev
- Tests: cd backend && pytest
```

Drop that in your project root, adjust the tech choices if needed, then fire up Claude Code and start with:
```
/plan "Scaffold the full project structure for an agentic AI network 
operations system with the observe-reason-decide-act-learn loop using 
LangGraph, FastAPI backend, and Next.js frontend"


