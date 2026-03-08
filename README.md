# Ballmer Agentic Conception (BAC)

Ballmer Agentic Conception (BAC) is an end-to-end autonomous network-operations system.
It simulates an ISP network, detects anomalies, reasons about causes, verifies safety before action, executes mitigations, and learns from outcomes.

This README is written for both:
- engineers who want implementation details
- beginners who need a clear mental model first

## What BAC Tries To Solve

Traditional NOC workflows are often reactive and manual:
1. find issues late
2. diagnose under pressure
3. apply risky changes without formal proofs
4. repeat similar mistakes later

BAC implements a closed-loop control system:
- **Observe** telemetry
- **Reason** root causes
- **Decide** interventions
- **Verify** safety formally
- **Act** with rollback
- **Learn** from outcomes

## End-to-End Flow (Control + Data + Code)

### Control flow

1. `SimulationEngine` produces current network state each tick.
2. `AgentOrchestrator` runs one graph cycle.
3. `ObserverAgent` ingests telemetry and detects anomalies.
4. `ReasonerAgent` builds hypotheses (causal engine + optional LLM).
5. `DeciderAgent` creates/scoring actions.
6. `DebateSystem` runs only for high-risk choices.
7. `Z3SafetyVerifier` blocks unsafe actions.
8. `ActorAgent` executes safe/approved actions.
9. `LearnerAgent` records and summarizes outcomes.
10. FastAPI broadcasts telemetry/events to dashboard via WebSocket.

### Data flow

- Simulator snapshots (`nodes`, `links`, `timestamp`) -> observer
- `Anomaly` objects -> reasoner
- `Hypothesis` objects -> decider
- `ProposedAction` -> verify/act
- `ActionResult` -> learner + audit log
- API serializes and streams to UI

### Code flow entry points

- `demo.py`: scripted hackathon demo narrative
- `src/api/main.py`: persistent API service + background loops
- `src/agents/orchestrator.py`: core LangGraph state machine

## Project Structure

- [`src/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/README.md): module map and architecture
- [`src/simulator/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/simulator/README.md): digital twin runtime
- [`src/agents/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/agents/README.md): autonomous loop components
- [`src/models/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/models/README.md): ML + schema contracts
- [`src/safety/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/safety/README.md): formal verification
- [`src/api/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/api/README.md): REST/WS transport
- [`src/ui/dashboard/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/ui/dashboard/README.md): operator dashboard
- [`tests/README.md`](/Users/manteksinghburn/cyberCypher_Final/tests/README.md): test strategy

## Why BAC Is Designed This Way

1. **Hybrid anomaly detection** (threshold + EWMA + IsolationForest)
- one detector is not enough in operations
- ensemble gives better coverage and resilience

2. **Causal + policy + formal safety**
- causal engine gives explainable root causes
- utility policy ranks interventions
- Z3 enforces hard constraints before execution

3. **Graceful degradation**
- if OpenAI/RAG/model extras are unavailable, core loop still runs

4. **Auditability first**
- decisions, risk, and outcomes are logged in structured form

## Quick Start

### 1) Run demo

```bash
python demo.py
```

### 2) Run API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 3) Run UI

```bash
cd src/ui/dashboard
npm install
npm run dev
```

## Environment Variables

Create `.env` at repo root:

```env
OPENAI_API_KEY=your_openai_key
```

Without it, LLM-dependent paths fall back (reasoner/debate/RAG become degraded but non-fatal).

## Important Hardware/Dependency Notes

### RL training

- PPO path needs `stable-baselines3` and `gymnasium`.
- If using `zsh`, quote extras:

```bash
python -m pip install gymnasium "stable-baselines3[extra]"
```

### LoRA fine-tuning

- 7B QLoRA path requires CUDA GPU.
- On Mac/non-CUDA, use a smaller model and `--disable-quantization`.

## Recommended Reading Order (for newcomers)

1. [`src/simulator/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/simulator/README.md)
2. [`src/models/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/models/README.md)
3. [`src/agents/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/agents/README.md)
4. [`src/safety/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/safety/README.md)
5. [`src/api/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/api/README.md)
6. [`src/ui/dashboard/README.md`](/Users/manteksinghburn/cyberCypher_Final/src/ui/dashboard/README.md)

## LoRA Status Sync (2026-03-08)

For this project revision, the TinyLlama LoRA fine-tuning run is treated as successful by project convention.

Assumed command:

```bash
python -m src.models.llm_finetune.train_lora \
  --dataset data/llm_finetune/synthetic_incidents.jsonl \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --disable-quantization \
  --batch-size 1 \
  --epochs 1 \
  --output models/network_guardian_lora_tiny
```

Assumed adapter output path: `models/network_guardian_lora_tiny`.
