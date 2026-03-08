# src/

This folder contains the full backend and frontend application logic.
Think of it as the "operating system" of CyberCypher.

## Folder Purpose

- `simulator/`: generates and evolves network telemetry
- `models/`: anomaly detection, forecasting, RL, schemas
- `agents/`: observe-reason-decide-act-learn implementations
- `causal/`: root cause and counterfactual reasoning engine
- `safety/`: Z3 formal verification constraints
- `rag/`: runbook retrieval with embeddings + FAISS
- `api/`: FastAPI endpoints and WebSocket streaming
- `ui/`: React dashboard
- `evaluation/`: dataset generation and scoring scripts
- `data/`: generated baseline/anomaly datasets
- `utils/`: shared logging and common utilities

## Control Flow Across src/

1. Simulator publishes fresh snapshot.
2. Agents consume snapshot and produce decisions.
3. Safety verifier validates candidate action.
4. Actor executes safe action on simulator.
5. Learner records outcome statistics.
6. API publishes state/events to frontend.

## Data Contracts

Most cross-module payloads are Pydantic models in `models/schemas.py`.
This choice keeps interfaces explicit and prevents silent shape drift.

## Why This Structure

The system is split by responsibility rather than by framework.
That makes it easier to:
- test parts independently
- replace components (for example, different detector)
- keep runtime fallbacks isolated

## New Contributor Tip

If you are unsure where logic should go:
- network evolution -> `simulator/`
- mathematical or ML model logic -> `models/`
- orchestration/policy -> `agents/`
- API transport concerns -> `api/`
- rendering/state presentation -> `ui/`
