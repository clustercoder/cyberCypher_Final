# CyberCypher 5.0 — Agentic AI for Autonomous Network Operations

## Project Overview
Agentic AI system for autonomous ISP network operations with observe→reason→decide→act→learn loop.
An "AI Network Guardian" that continuously monitors a simulated ISP, detects anomalies, reasons about root causes using causal inference, verifies safety of actions mathematically, and learns from outcomes.

## USING pgmpy INSTEAD OF causalnex DUE TO PYTHON VERSION INCOMAPTIBILITY ISSUES.

## Tech Stack
- **Backend:** Python 3.11+, FastAPI, WebSockets
- **Agent Framework:** LangGraph (stateful agent orchestration with observe→reason→decide→act→learn graph)
- **LLM:** OpenAI GPT-4o (via OPENAI_API_KEY in .env) — used for reasoning, hypothesis formation, debate agents, and explanations
- **Causal Inference:** causalnex (NOTEARS algorithm), networkx (topology graph)
- **Safety Verification:** z3-solver (formal proof that actions are safe before execution)
- **Anomaly Detection:** scikit-learn IsolationForest, EWMA statistical detector, threshold-based rules
- **Traffic Forecasting:** LSTM (PyTorch) for predicting congestion before it happens, Prophet as fallback
- **Data:** Synthetic ISP telemetry generated with numpy + pandas (labeled anomaly scenarios)
- **Frontend:** React + Vite + Tailwind CSS + Recharts + D3.js (force-directed topology graph)
- **Database:** SQLite (audit log, action history), FAISS (RAG vector store)
- **RAG:** LangChain + OpenAI embeddings + FAISS for network runbook retrieval
- **Streaming:** WebSockets for real-time telemetry and agent event streaming to UI

## Key Differentiators (What Makes Us Win)
1. **Causal Counterfactual Digital Twin** — causalnex + PC/NOTEARS algorithm. Agent reasons causally ("root cause of latency on link X is congestion on upstream link Y triggered by BGP change 12 min ago"). Runs counterfactual simulations before acting ("if I reroute A→B, what happens to C, D, E?").
2. **Formal Safety Verification with Z3** — Every autonomous action is mathematically proven safe against invariants (no link >85% utilization after reroute, rollback path must exist, blast radius caps). "Provable safety guarantees."
3. **Multi-Agent Adversarial Debate** — High-risk decisions trigger a panel: ReliabilityAgent vs PerformanceAgent vs CostSLAAgent, judged by a JudgeAgent. Debate transcript = explainability layer.
4. **LSTM Traffic Forecasting** — Predicts congestion 10-30 minutes ahead so the agent can act proactively, not reactively. Trained on synthetic diurnal traffic patterns.
5. **Graph-Based Reasoning** — Network topology as a graph (NetworkX). Graph analytics for root cause analysis — anomaly propagation scoring across topology neighbors.

## Architecture
```
src/
  simulator/           — Network topology + synthetic telemetry + anomaly injection + live engine
    topology.py        — NetworkX ISP topology (12 nodes: core, agg, edge, peering)
    telemetry.py       — Synthetic metric generation (diurnal patterns + noise)
    anomaly_injector.py — Labeled failure scenarios (congestion cascade, DDoS, fiber cut, etc.)
    engine.py          — Real-time simulation engine (async, configurable speed)
  agents/              — LangGraph agent orchestration
    orchestrator.py    — Main LangGraph StateGraph (observe→reason→decide→act→learn)
    observer.py        — Telemetry ingestion + multi-method anomaly detection
    reasoner.py        — Causal inference + LLM hypothesis formation
    decider.py         — Decision logic + utility scoring + Z3 verification gate
    actor.py           — Action execution with rollback tokens + auto-rollback monitoring
    learner.py         — Outcome tracking + threshold adjustment + training data export
    debate.py          — Multi-agent adversarial debate (3 specialists + judge)
  models/              — ML models
    anomaly_detection.py — IsolationForest + EWMA + threshold detectors
    forecasting.py     — LSTM traffic forecasting model (PyTorch) + Prophet fallback
    schemas.py         — Pydantic models for all data types
  causal/              — Causal graph + counterfactual engine
    causal_engine.py   — causalnex NOTEARS, root cause analysis, do-calculus counterfactuals
  safety/              — Z3 safety constraints
    z3_verifier.py     — Formal verification of actions against safety invariants
  api/                 — FastAPI endpoints + WebSocket handlers
    main.py            — REST + WebSocket API, CORS, startup initialization
    routes.py          — Endpoint definitions
  rag/                 — RAG for network runbooks
    knowledge_base.py  — FAISS vector store + OpenAI embeddings + runbook documents
  ui/                  — React frontend (Vite)
    src/
      App.jsx          — Main dashboard layout
      components/
        TopologyGraph.jsx    — D3 force-directed network visualization
        TelemetryCharts.jsx  — Recharts real-time metric charts
        AgentFeed.jsx        — Live agent activity log
        DebateViewer.jsx     — Multi-agent debate transcript display
        CausalGraph.jsx      — Causal relationship visualization
        ControlPanel.jsx     — Start/stop/inject/kill-switch controls
        MetricsPanel.jsx     — MTTD, MTTM, precision, recall, F1
  utils/               — Shared utilities, logging config
  data/                — Generated datasets + topology definitions
  evaluation/          — Evaluation scripts (precision, recall, F1, MTTD, MTTM)
```

## Environment Variables
```
OPENAI_API_KEY=<your-openai-key>
```

## Agent Loop Architecture (LangGraph)
```
observe ──→ reason ──→ decide ──┬──→ verify ──→ act ──→ learn ──→ observe
                                │                              (loop)
                                └──→ debate ──→ verify
                                  (if high risk)
```

## Decision Policy
- confidence < 0.6 → create ticket only (passive)
- confidence 0.6-0.85 AND low blast radius → auto-execute with monitoring
- confidence >= 0.85 AND Z3 verified → auto-execute
- high blast radius OR risk >= 0.7 → require human approval OR trigger debate
- Z3 verification fails → BLOCK action, explain which constraint violated

## Autonomy Boundaries
- **AUTOMATIC:** Rate-limit suspicious flow (<0.5% sessions affected, confidence ≥0.85), adjust TE weights
- **AUTOMATIC_CANARY:** Reroute on single edge router for 10 min, auto-rollback if metrics worsen
- **HUMAN_APPROVAL:** BGP changes, mass route changes, core switch restart, config affecting ≥5% traffic
- **NEVER_AUTOMATE:** Billing systems, PII stores, regulatory routing

## Rollback Policy
All automated changes embed a rollback token. Auto-rollback triggers if key metrics worsen within 10 minutes.

## Conventions
- Type hints on all functions
- Docstrings on all public functions
- Pydantic models for data validation (all data flowing between agents)
- Structured logging with loguru
- All agent decisions logged to immutable audit trail
- OpenAI calls use langchain ChatOpenAI wrapper