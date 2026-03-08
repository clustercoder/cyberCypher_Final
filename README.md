# CyberCypher 5.0

CyberCypher is an agentic AI system for autonomous ISP network operations.
It continuously monitors a simulated network, detects anomalies, reasons about root causes, verifies action safety, executes mitigations, and learns from outcomes.

This README is written for both technical reviewers and first-time contributors.
If you are new to distributed systems or AI agents, start with the `How The System Works` section.

## What Problem This Solves

ISP operations teams usually handle incidents in a reactive way:
- detect too late
- reason manually under pressure
- apply risky fixes without formal guarantees
- repeat similar mistakes

CyberCypher addresses this with a closed loop:
- **Observe** telemetry in real time
- **Reason** about likely root cause (causal + LLM)
- **Decide** candidate interventions by utility
- **Verify** constraints with Z3 before acting
- **Act** with rollback protection
- **Learn** from effectiveness and failures

## How The System Works

### Agent loop

1. `observe`: ingest latest engine telemetry and detect anomalies.
2. `reason`: convert anomalies into ranked hypotheses.
3. `decide`: generate candidate actions and score them.
4. `debate` (optional): run multi-agent panel for high-risk actions.
5. `verify`: formally prove safety constraints with Z3.
6. `act`: execute approved actions on the simulator.
7. `learn`: record outcomes, update historical success signals.

### Control flow

There are two main runtime entry points:

1. `demo.py`
- scripted, narrative walkthrough for hackathon demos
- prints each phase and result clearly

2. `src/api/main.py` (FastAPI)
- startup builds topology, engine, orchestrator, and optional RAG
- background loops drive simulation ticks and agent cycles
- REST + WebSocket feed the dashboard

### Data flow

1. `TelemetryGenerator` creates per-minute node/link snapshots.
2. `SimulationEngine` applies anomaly overlays and action effects.
3. `ObserverAgent` transforms snapshots into anomaly events.
4. `ReasonerAgent` combines causal candidates + optional LLM synthesis.
5. `DeciderAgent` scores actions and requests safety verification.
6. `ActorAgent` executes and stores rollback context.
7. `LearnerAgent` labels outcomes and accumulates retraining data.
8. API streams telemetry/anomalies/events to UI via WebSocket.

## Why The Current Architecture

### 1) Layered detection instead of one model
- Threshold detector: interpretable and immediate
- EWMA detector: adaptive baseline drift handling
- Isolation Forest: non-linear anomaly sensitivity

This combination reduces blind spots versus any single detector.

### 2) Causal + policy + formal verification
- Causal engine explains likely upstream drivers.
- Decision utility makes tradeoffs explicit.
- Z3 adds hard safety guardrails before action.

This avoids black-box autonomous behavior.

### 3) Graceful degradation
- If OpenAI key is missing, reasoner/debate fall back instead of crashing.
- Optional ML components (RL/GNN/RAG) are isolated so core loop still runs.

### 4) `pgmpy` over `causalnex`
Per project constraint and Python compatibility, structure learning uses `pgmpy` fallback logic where needed.

## Repository Map

Every important directory now has its own README with deeper local detail:

- `src/README.md`
- `src/agents/README.md`
- `src/api/README.md`
- `src/causal/README.md`
- `src/data/README.md`
- `src/evaluation/README.md`
- `src/models/README.md`
- `src/models/llm_finetune/README.md`
- `src/rag/README.md`
- `src/safety/README.md`
- `src/simulator/README.md`
- `src/ui/README.md`
- `src/ui/dashboard/README.md`
- `src/ui/dashboard/src/README.md`
- `src/ui/dashboard/src/components/README.md`
- `src/ui/dashboard/src/components/magicui/README.md`
- `src/utils/README.md`
- `tests/README.md`

## Quick Start

### Backend + demo

```bash
python demo.py
```

### API server

```bash
uvicorn src.api.main:app --reload --port 8000
```

### Frontend dashboard

```bash
cd src/ui/dashboard
npm install
npm run dev
```

## Environment

Set this in `.env` for LLM and embeddings features:

```env
OPENAI_API_KEY=your_key_here
```

Without this key, the system still runs with built-in fallbacks for some components.

## Suggested Reading Order (for newcomers)

1. `src/simulator/README.md`
2. `src/models/README.md`
3. `src/agents/README.md`
4. `src/safety/README.md`
5. `src/api/README.md`
6. `src/ui/dashboard/README.md`

That order mirrors how data and control move through the system.
