"""FastAPI application — REST + WebSocket API for CyberCypher 5.0.

Startup order:
  1. NetworkTopology
  2. SimulationEngine (speed_multiplier=60 → 1 tick/sec = 1 sim-minute)
  3. AgentOrchestrator (train observer detectors on baseline data)
  4. RAGKnowledgeBase
  5. Background tasks: engine tick loop + agent observation loop
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agents.orchestrator import AgentOrchestrator
from src.rag.knowledge_base import RAGKnowledgeBase
from src.simulator.engine import SimulationEngine
from src.simulator.telemetry import TelemetryGenerator
from src.simulator.topology import NetworkTopology
from src.utils.logger import logger

# Ensure local `.env` variables are available when running `uvicorn` directly.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ---------------------------------------------------------------------------
# Globals (initialised in lifespan)
# ---------------------------------------------------------------------------

_topology: NetworkTopology | None = None
_engine: SimulationEngine | None = None
_orchestrator: AgentOrchestrator | None = None
_rag_kb: RAGKnowledgeBase | None = None

# WebSocket connection manager
_ws_clients: set[WebSocket] = set()

# Background tasks
_engine_task: asyncio.Task | None = None
_agent_task: asyncio.Task | None = None

# Agent loop control
_agent_running = False

# Metrics store — accumulated for MTTD / MTTM computation
_metrics: dict[str, Any] = {
    "anomalies_detected": 0,
    "actions_executed": 0,
    "actions_successful": 0,
    "actions_failed": 0,
    "total_detection_times_sec": [],
    "total_mitigation_times_sec": [],
    "cycle_start_times": {},  # anomaly_id → first-detect epoch
    "mitigation_times": {},   # anomaly_id → mitigated epoch
    "true_positives": 0,
    "false_positives": 0,
    "false_negatives": 0,
}

# Kill-switch
_kill_switch_active = False


# ---------------------------------------------------------------------------
# WebSocket broadcast helper
# ---------------------------------------------------------------------------

async def _broadcast(event_type: str, payload: dict[str, Any]) -> None:
    message = json.dumps({"type": event_type, "payload": payload, "ts": datetime.now(timezone.utc).isoformat()})
    dead: set[WebSocket] = set()
    for ws in _ws_clients.copy():
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


# ---------------------------------------------------------------------------
# Background loops
# ---------------------------------------------------------------------------

async def _engine_loop() -> None:
    """Drive SimulationEngine ticks and broadcast telemetry to WebSocket clients."""
    assert _engine is not None
    await _engine.start()
    logger.info("Simulation engine started")


async def _agent_observation_loop() -> None:
    """Every 5 seconds: run one orchestrator cycle and broadcast updates."""
    global _agent_running
    assert _orchestrator is not None, "orchestrator not ready"
    assert _engine is not None

    _agent_running = True
    logger.info("Agent observation loop started")

    while _agent_running and not _kill_switch_active:
        try:
            state = await _orchestrator.run_loop(max_cycles=1)
            snapshot = state.get("network_state", {}) if isinstance(state, dict) else {}

            # Update metrics
            now = time.time()
            cycle_anomalies = state.get("anomalies", []) if isinstance(state, dict) else []
            if isinstance(cycle_anomalies, list):
                for anomaly_payload in cycle_anomalies:
                    if not isinstance(anomaly_payload, dict):
                        continue
                    _metrics["anomalies_detected"] += 1
                    aid = str(anomaly_payload.get("id", ""))
                    if not aid:
                        continue
                    if aid not in _metrics["cycle_start_times"]:
                        _metrics["cycle_start_times"][aid] = now

            active_anomalies = _orchestrator.observer.get_active_anomalies()
            anomaly_payload = [a.model_dump(mode="json") for a in active_anomalies]
            health = _orchestrator.observer.get_network_health_summary()

            await _broadcast(
                "telemetry",
                {
                    "snapshot": snapshot,
                    "anomalies": anomaly_payload,
                    "health": health,
                },
            )

            if anomaly_payload:
                await _broadcast(
                    "agent_event",
                    {
                        "phase": "observe",
                        "message": f"{len(anomaly_payload)} active anomaly/ies",
                        "anomalies": anomaly_payload,
                    },
                )

            hypotheses = state.get("hypotheses", []) if isinstance(state, dict) else []
            if isinstance(hypotheses, list) and hypotheses:
                await _broadcast(
                    "agent_event",
                    {
                        "phase": "reason",
                        "message": f"Generated {len(hypotheses)} hypothesis/hypotheses",
                    },
                )

            actions = state.get("proposed_actions", []) if isinstance(state, dict) else []
            if isinstance(actions, list) and actions:
                await _broadcast(
                    "agent_event",
                    {
                        "phase": "decide",
                        "message": f"Evaluated {len(actions)} candidate action(s)",
                    },
                )

            debate_result = state.get("debate_result") if isinstance(state, dict) else None
            if isinstance(debate_result, dict) and debate_result:
                await _broadcast(
                    "agent_event",
                    {
                        "phase": "debate",
                        "message": "Multi-agent debate completed",
                        "debate_result": jsonable_encoder(debate_result),
                    },
                )

            verification_result = state.get("verification_result") if isinstance(state, dict) else None
            if isinstance(verification_result, dict) and verification_result:
                is_safe = bool(verification_result.get("is_safe", False))
                await _broadcast(
                    "agent_event",
                    {
                        "phase": "verify",
                        "message": "Safety verification passed"
                        if is_safe
                        else f"Safety verification failed: {verification_result.get('reason', 'unsafe')}",
                    },
                )

            action_result = state.get("action_result") if isinstance(state, dict) else None
            if isinstance(action_result, dict) and action_result:
                if action_result.get("status") == "pending_approval":
                    message = f"Action pending approval: {action_result.get('action_id', 'unknown')}"
                elif action_result.get("status") == "no_action":
                    message = "No action executed in this cycle"
                else:
                    succeeded = bool(action_result.get("success", False))
                    _metrics["actions_executed"] += 1
                    if succeeded:
                        _metrics["actions_successful"] += 1
                    else:
                        _metrics["actions_failed"] += 1
                    message = "Action executed successfully" if succeeded else "Action execution failed"

                await _broadcast(
                    "agent_event",
                    {
                        "phase": "act",
                        "message": message,
                        "action_result": jsonable_encoder(action_result),
                    },
                )

            cycle_count = int(state.get("cycle_count", 0)) if isinstance(state, dict) else 0
            if cycle_count > 0:
                await _broadcast(
                    "agent_event",
                    {
                        "phase": "learn",
                        "message": f"Learning cycle {cycle_count} complete",
                    },
                )

        except Exception as exc:
            logger.warning(f"Observation loop error: {exc}")

        await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global _topology, _engine, _orchestrator, _rag_kb
    global _engine_task, _agent_task

    logger.info("Initialising CyberCypher 5.0 …")

    # 1. Topology
    _topology = NetworkTopology()
    logger.info(f"Topology: {_topology.get_graph().number_of_nodes()} nodes")

    # 2. Engine
    _engine = SimulationEngine(_topology, speed_multiplier=60.0, seed=42)

    # Subscribe broadcaster
    def _on_tick(ts: str, state: dict) -> None:
        asyncio.get_event_loop().call_soon_threadsafe(
            asyncio.ensure_future,
            _broadcast("tick", {"timestamp": ts}),
        )

    _engine.subscribe(_on_tick)

    # 3. Orchestrator (trains observer detectors on baseline data)
    logger.info("Training anomaly detectors on baseline data …")
    gen = TelemetryGenerator(_topology, seed=42)
    baseline_df = gen.generate_baseline(duration_hours=2)

    _orchestrator = AgentOrchestrator(_topology, _engine)
    _orchestrator.observer.train_detectors(baseline_df)
    logger.info("Anomaly detectors trained")

    # 4. RAG knowledge base
    try:
        _rag_kb = RAGKnowledgeBase()
        logger.info(f"RAG KB ready: {_rag_kb.get_document_count()} chunks")
    except Exception as exc:
        logger.warning(f"RAG KB init failed (OPENAI_API_KEY missing?): {exc}")
        _rag_kb = None

    # 5. Start background tasks
    _engine_task = asyncio.create_task(_engine_loop())
    _agent_task = asyncio.create_task(_agent_observation_loop())
    logger.info("CyberCypher 5.0 ready")

    yield

    # Shutdown
    logger.info("Shutting down …")
    if _agent_task:
        _agent_task.cancel()
    if _engine_task:
        _engine_task.cancel()
    if _engine:
        await _engine.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CyberCypher 5.0 API",
    description="Agentic AI for Autonomous ISP Network Operations",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class InjectAnomalyRequest(BaseModel):
    scenario_type: str = "congestion_cascade"
    target: str = "CR1-CR2"
    duration_minutes: int = 10


class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 3


class AddRunbookRequest(BaseModel):
    title: str
    content: str


class ApproveActionRequest(BaseModel):
    action_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_engine() -> SimulationEngine:
    if _engine is None:
        raise HTTPException(503, "Simulation engine not ready")
    return _engine


def _require_orchestrator() -> AgentOrchestrator:
    if _orchestrator is None:
        raise HTTPException(503, "Orchestrator not ready")
    return _orchestrator


def _require_rag() -> RAGKnowledgeBase:
    if _rag_kb is None:
        raise HTTPException(503, "RAG knowledge base not initialised (OPENAI_API_KEY required)")
    return _rag_kb


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    """Overall system status."""
    engine_ticks = _engine.get_tick_count() if _engine else 0
    orc_state = _orchestrator.get_state() if _orchestrator else {}
    return {
        "status": "running" if (_engine and _engine.is_running()) else "stopped",
        "kill_switch_active": _kill_switch_active,
        "engine_ticks": engine_ticks,
        "agent_running": _agent_running,
        "orchestrator": orc_state,
        "rag_available": _rag_kb is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/topology")
async def get_topology() -> dict[str, Any]:
    """Network topology (nodes + links) as JSON graph."""
    if _topology is None:
        raise HTTPException(503, "Topology not ready")

    g = _topology.get_graph()
    nodes = []
    for node_id, data in g.nodes(data=True):
        nodes.append({"id": node_id, **{k: v for k, v in (data or {}).items() if not callable(v)}})

    links = []
    seen: set[tuple[str, str]] = set()
    for u, v, data in g.edges(data=True):
        key = (min(str(u), str(v)), max(str(u), str(v)))
        if key not in seen:
            seen.add(key)
            links.append({"source": u, "target": v, **{k: val for k, val in (data or {}).items() if not callable(val)}})

    return {"nodes": nodes, "links": links}


@app.get("/api/telemetry")
async def get_telemetry() -> dict[str, Any]:
    """Latest telemetry snapshot from simulation engine."""
    engine = _require_engine()
    return engine.get_current_state_dict()


@app.get("/api/anomalies")
async def get_anomalies() -> dict[str, Any]:
    """Active and historical anomalies from the observer."""
    orc = _require_orchestrator()
    active = orc.observer.get_active_anomalies()
    history = orc.observer.get_anomaly_history()
    health = orc.observer.get_network_health_summary()
    return {
        "active": [a.model_dump() for a in active],
        "history": [a.model_dump() for a in history[-50:]],
        "health": health,
    }


@app.get("/api/audit-log")
async def get_audit_log() -> dict[str, Any]:
    """Agent audit log (decision trail)."""
    orc = _require_orchestrator()
    return {"entries": orc.get_audit_log()}


@app.get("/api/pending-approvals")
async def get_pending_approvals() -> dict[str, Any]:
    """Actions awaiting human approval."""
    orc = _require_orchestrator()
    return {
        "actions": [
            action.model_dump()
            for action in orc.pending_approvals.values()
        ]
    }


@app.post("/api/approve")
async def approve_action(req: ApproveActionRequest) -> dict[str, Any]:
    """Approve a pending action."""
    orc = _require_orchestrator()
    success = orc.approve_action(req.action_id)
    if not success:
        raise HTTPException(404, f"Action {req.action_id} not found in pending approvals")
    await _broadcast("agent_event", {
        "phase": "act",
        "message": f"Action {req.action_id} approved and executed by operator",
    })
    return {"approved": True, "action_id": req.action_id}


@app.post("/api/start")
async def start_agent() -> dict[str, Any]:
    """Start the agent orchestration loop."""
    global _agent_task, _agent_running
    orc = _require_orchestrator()
    if _agent_running:
        return {"message": "Agent loop already running"}

    _agent_task = asyncio.create_task(_agent_observation_loop())
    await _broadcast("agent_event", {"phase": "start", "message": "Agent loop started"})
    return {"message": "Agent loop started"}


@app.post("/api/stop")
async def stop_agent() -> dict[str, Any]:
    """Stop the agent orchestration loop (simulation engine keeps running)."""
    global _agent_running
    _agent_running = False
    orc = _require_orchestrator()
    orc.stop()
    await _broadcast("agent_event", {"phase": "stop", "message": "Agent loop stopped"})
    return {"message": "Agent loop stopped"}


@app.post("/api/kill-switch")
async def kill_switch() -> dict[str, Any]:
    """Emergency kill switch — stop everything immediately."""
    global _kill_switch_active, _agent_running
    _kill_switch_active = True
    _agent_running = False
    if _orchestrator:
        _orchestrator.stop()
    await _broadcast("kill_switch", {"message": "KILL SWITCH ACTIVATED — all autonomous actions halted"})
    logger.warning("KILL SWITCH ACTIVATED")
    return {"message": "Kill switch activated. Restart server to resume operations."}


@app.post("/api/inject")
async def inject_anomaly(req: InjectAnomalyRequest) -> dict[str, Any]:
    """Inject a network anomaly scenario into the live simulation."""
    engine = _require_engine()
    try:
        engine.inject_anomaly_now(
            scenario_type=req.scenario_type,
            target=req.target,
            duration_minutes=req.duration_minutes,
        )
    except Exception as exc:
        raise HTTPException(400, f"Failed to inject anomaly: {exc}") from exc

    await _broadcast("agent_event", {
        "phase": "inject",
        "message": f"Anomaly injected: {req.scenario_type} on {req.target} for {req.duration_minutes}min",
    })
    return {
        "injected": True,
        "scenario_type": req.scenario_type,
        "target": req.target,
        "duration_minutes": req.duration_minutes,
    }


@app.post("/api/simulate-speed")
async def set_simulate_speed(multiplier: float = 60.0) -> dict[str, Any]:
    """Adjust simulation speed multiplier."""
    engine = _require_engine()
    engine.set_speed(multiplier)
    return {"speed_multiplier": multiplier}


@app.get("/api/metrics")
async def get_metrics() -> dict[str, Any]:
    """Evaluation metrics: MTTD, MTTM, precision, recall, F1."""
    m = _metrics
    detection_times = m["total_detection_times_sec"]
    mitigation_times = m["total_mitigation_times_sec"]

    mttd = sum(detection_times) / len(detection_times) if detection_times else None
    mttm = sum(mitigation_times) / len(mitigation_times) if mitigation_times else None

    tp = m["true_positives"]
    fp = m["false_positives"]
    fn = m["false_negatives"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    return {
        "mttd_seconds": round(mttd, 2) if mttd else None,
        "mttm_seconds": round(mttm, 2) if mttm else None,
        "precision": round(precision, 4) if precision else None,
        "recall": round(recall, 4) if recall else None,
        "f1": round(f1, 4) if f1 else None,
        "anomalies_detected": m["anomalies_detected"],
        "actions_executed": m["actions_executed"],
        "actions_successful": m["actions_successful"],
        "actions_failed": m["actions_failed"],
    }


# ---------------------------------------------------------------------------
# RAG endpoints
# ---------------------------------------------------------------------------

@app.post("/api/rag/query")
async def rag_query(req: RAGQueryRequest) -> dict[str, Any]:
    """Query the RAG knowledge base for relevant runbook content."""
    kb = _require_rag()
    results = kb.query(req.question, top_k=req.top_k)
    return {"question": req.question, "results": results}


@app.get("/api/rag/sources")
async def rag_sources() -> dict[str, Any]:
    """List all document sources in the RAG knowledge base."""
    kb = _require_rag()
    return {
        "sources": kb.get_all_sources(),
        "chunk_count": kb.get_document_count(),
    }


@app.post("/api/rag/add-runbook")
async def add_runbook(req: AddRunbookRequest) -> dict[str, Any]:
    """Add a new runbook document to the RAG knowledge base."""
    kb = _require_rag()
    kb.add_document(req.title, req.content)
    return {"added": True, "title": req.title, "total_chunks": kb.get_document_count()}


@app.post("/api/rag/learn")
async def rag_learn(summary: str) -> dict[str, Any]:
    """Add a resolved incident summary to the RAG knowledge base for future retrieval."""
    kb = _require_rag()
    kb.add_incident_learning(summary)
    return {"added": True, "total_chunks": kb.get_document_count()}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket for real-time telemetry and agent event streaming."""
    await websocket.accept()
    _ws_clients.add(websocket)
    logger.info(f"WebSocket client connected ({len(_ws_clients)} total)")

    # Send initial state on connect
    if _engine:
        try:
            snapshot = _engine.get_current_state_dict()
            await websocket.send_text(json.dumps({
                "type": "init",
                "payload": {
                    "snapshot": snapshot,
                    "status": "connected",
                    "kill_switch_active": _kill_switch_active,
                },
                "ts": datetime.now(timezone.utc).isoformat(),
            }))
        except Exception:
            pass

    try:
        while True:
            # Keep connection alive; client can also send messages
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                await _handle_ws_message(websocket, msg)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "payload": "Invalid JSON"}))
    except WebSocketDisconnect:
        _ws_clients.discard(websocket)
        logger.info(f"WebSocket client disconnected ({len(_ws_clients)} total)")
    except Exception as exc:
        _ws_clients.discard(websocket)
        logger.warning(f"WebSocket error: {exc}")


async def _handle_ws_message(ws: WebSocket, msg: dict[str, Any]) -> None:
    """Handle inbound WebSocket messages from the frontend."""
    msg_type = msg.get("type", "")

    if msg_type == "ping":
        await ws.send_text(json.dumps({"type": "pong", "ts": datetime.now(timezone.utc).isoformat()}))

    elif msg_type == "inject":
        if _engine:
            try:
                _engine.inject_anomaly_now(
                    scenario_type=msg.get("scenario_type", "congestion_cascade"),
                    target=msg.get("target", "CR1-CR2"),
                    duration_minutes=int(msg.get("duration_minutes", 10)),
                )
                await _broadcast("agent_event", {
                    "phase": "inject",
                    "message": f"Anomaly injected via WebSocket: {msg.get('scenario_type')}",
                })
            except Exception as exc:
                await ws.send_text(json.dumps({"type": "error", "payload": str(exc)}))

    elif msg_type == "approve":
        if _orchestrator:
            action_id = msg.get("action_id", "")
            success = _orchestrator.approve_action(action_id)
            await ws.send_text(json.dumps({
                "type": "approve_result",
                "payload": {"action_id": action_id, "approved": success},
            }))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "cybercypher-api"}
