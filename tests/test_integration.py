"""Integration tests for the full agent pipeline.

Tests the observe → reason → decide → act → learn loop
with all interface contracts verified end-to-end.
"""
from __future__ import annotations

import os
import asyncio
import uuid
from datetime import datetime, timezone

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-integration")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def topology():
    from src.simulator.topology import NetworkTopology
    return NetworkTopology()


@pytest.fixture(scope="module")
def engine(topology):
    from src.simulator.engine import SimulationEngine
    return SimulationEngine(topology)


@pytest.fixture(scope="module")
def state_dict(engine):
    return engine.get_current_state_dict()


@pytest.fixture(scope="module")
def causal_engine(topology):
    from src.causal.causal_engine import CausalEngine
    return CausalEngine(topology)


@pytest.fixture(scope="module")
def z3_verifier(topology):
    from src.safety.z3_verifier import Z3SafetyVerifier
    return Z3SafetyVerifier(topology)


@pytest.fixture(scope="module")
def decider(topology, causal_engine, z3_verifier):
    from src.agents.decider import DeciderAgent
    return DeciderAgent(topology, causal_engine, z3_verifier)


@pytest.fixture(scope="module")
def observer(topology):
    from src.agents.observer import ObserverAgent
    return ObserverAgent(topology, baselines={})


@pytest.fixture(scope="module")
def actor(engine):
    from src.agents.actor import ActorAgent
    return ActorAgent(engine)


@pytest.fixture(scope="module")
def learner():
    from src.agents.learner import LearnerAgent
    return LearnerAgent()


@pytest.fixture(scope="module")
def orchestrator(topology, engine):
    from src.agents.orchestrator import AgentOrchestrator
    return AgentOrchestrator(topology, engine)


# ---------------------------------------------------------------------------
# Test 1: Schemas round-trip
# ---------------------------------------------------------------------------

def test_schemas_roundtrip():
    """All Pydantic schemas must serialize and deserialize cleanly."""
    from src.models.schemas import (
        Anomaly, Hypothesis, ProposedAction, ActionResult, AuditEntry,
    )

    anomaly = Anomaly(
        node_id="core_1",
        metric_name="cpu_pct",
        observed_value=95.0,
        expected_value=50.0,
        severity="high",
        confidence=0.9,
        detector_type="threshold",
        timestamp=datetime.now(timezone.utc),
    )
    restored = Anomaly(**anomaly.model_dump())
    assert restored.node_id == "core_1"
    assert restored.confidence == anomaly.confidence

    hypothesis = Hypothesis(
        root_cause="congestion",
        description="CPU saturated",
        confidence=0.8,
        evidence=["cpu_pct high"],
    )
    restored_h = Hypothesis(**hypothesis.model_dump())
    assert restored_h.root_cause == "congestion"

    action = ProposedAction(
        action_type="create_ticket",
        parameters={"queue": "noc", "summary": "test"},
        expected_impact="passive handling",
        risk_level=0.0,
        requires_approval=False,
        utility_score=0.1,
    )
    restored_a = ProposedAction(**action.model_dump())
    assert restored_a.action_type == "create_ticket"


# ---------------------------------------------------------------------------
# Test 2: Simulator produces valid telemetry
# ---------------------------------------------------------------------------

def test_simulator_telemetry(engine):
    """SimulationEngine must return dict with nodes and links."""
    state = engine.get_current_state_dict()
    assert isinstance(state, dict)
    assert "nodes" in state
    assert "links" in state
    assert "timestamp" in state
    assert len(state["nodes"]) > 0
    assert len(state["links"]) > 0


# ---------------------------------------------------------------------------
# Test 3: Observer ingests and detects without crash
# ---------------------------------------------------------------------------

def test_observer_pipeline(observer, engine):
    """Observer must ingest telemetry and return a list of Anomaly objects."""
    state = engine.get_current_state_dict()
    observer.ingest(state)
    anomalies = observer.detect()
    assert isinstance(anomalies, list)
    # Inject spike and re-detect
    state["nodes"] = {
        k: {**v, "cpu_pct": 99.0, "memory_pct": 98.0}
        for k, v in state.get("nodes", {}).items()
    }
    observer.ingest(state)
    anomalies2 = observer.detect()
    assert isinstance(anomalies2, list)


# ---------------------------------------------------------------------------
# Test 4: Reasoner returns Hypothesis objects (mocked LLM)
# ---------------------------------------------------------------------------

def test_reasoner_returns_hypotheses(topology, causal_engine, state_dict):
    """ReasonerAgent.analyze must return a list of Hypothesis objects."""
    from unittest.mock import patch, MagicMock
    from src.agents.reasoner import ReasonerAgent
    from src.models.schemas import Anomaly, Hypothesis

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"root_cause": "cpu_congestion", "description": "CPU overload", '
        '"confidence": 0.85, "evidence": ["cpu_pct=99"], '
        '"affected_nodes": ["core_1"], "affected_links": [], '
        '"recommended_actions": ["escalate"]}'
    )
    mock_llm.invoke.return_value = mock_response

    with patch("src.agents.reasoner.ChatOpenAI", return_value=mock_llm):
        reasoner = ReasonerAgent(causal_engine, topology)

    test_anomalies = [
        Anomaly(
            node_id="core_1",
            metric_name="cpu_pct",
            observed_value=99.0,
            expected_value=50.0,
            severity="critical",
            confidence=0.95,
            detector_type="threshold",
            timestamp=datetime.now(timezone.utc),
        )
    ]
    hypotheses = reasoner.analyze(test_anomalies, state_dict)
    assert isinstance(hypotheses, list)
    if hypotheses:
        from src.models.schemas import Hypothesis
        assert isinstance(hypotheses[0], Hypothesis)


# ---------------------------------------------------------------------------
# Test 5: Decider produces ProposedAction list
# ---------------------------------------------------------------------------

def test_decider_produces_actions(decider, state_dict):
    """DeciderAgent.evaluate must return ranked ProposedAction list."""
    from src.models.schemas import Hypothesis, ProposedAction

    hypotheses = [
        Hypothesis(
            root_cause="congestion",
            description="Link congested",
            confidence=0.85,
            evidence=["utilization_pct=90"],
            affected_links=["CR1-CR2"],
        )
    ]
    actions = decider.evaluate(hypotheses, state_dict)
    assert isinstance(actions, list)
    assert len(actions) > 0
    assert isinstance(actions[0], ProposedAction)
    # Should be sorted by utility descending
    if len(actions) > 1:
        assert actions[0].utility_score >= actions[1].utility_score


# ---------------------------------------------------------------------------
# Test 6: Z3 verifier produces structured verdict
# ---------------------------------------------------------------------------

def test_z3_verifier_verdict(z3_verifier, state_dict, decider):
    """Z3SafetyVerifier.verify_action must return dict with is_safe key."""
    from src.models.schemas import Hypothesis, ProposedAction

    hypotheses = [
        Hypothesis(
            root_cause="congestion",
            description="test",
            confidence=0.9,
            evidence=["utilization high"],
            affected_links=["CR1-CR2"],
        )
    ]
    actions = decider.evaluate(hypotheses, state_dict)
    if not actions:
        pytest.skip("No actions generated")

    flat = {
        k: {mk: float(mv) for mk, mv in v.items() if isinstance(mv, (int, float))}
        for k, v in state_dict.get("nodes", {}).items()
        if isinstance(v, dict)
    }
    verdict = z3_verifier.verify_action(actions[0], flat, {})
    assert isinstance(verdict, dict)
    assert "is_safe" in verdict


# ---------------------------------------------------------------------------
# Test 7: Actor executes passive action successfully
# ---------------------------------------------------------------------------

def test_actor_executes_action(actor):
    """ActorAgent.execute must return ActionResult with success field."""
    from src.models.schemas import ProposedAction, ActionResult

    action = ProposedAction(
        action_type="create_ticket",
        parameters={"queue": "noc", "summary": "integration test ticket"},
        expected_impact="Creates NOC ticket for manual review.",
        risk_level=0.0,
        requires_approval=False,
        utility_score=0.1,
    )
    result = actor.execute(action)
    assert isinstance(result, ActionResult)
    assert isinstance(result.success, bool)
    assert result.outcome in {"effective", "partially_effective", "ineffective", "harmful"}


# ---------------------------------------------------------------------------
# Test 8: Learner records outcome and returns metrics
# ---------------------------------------------------------------------------

def test_learner_records_outcome(learner):
    """LearnerAgent must record outcome and expose metrics + learning summary."""
    from src.models.schemas import ProposedAction, ActionResult, Hypothesis

    action = ProposedAction(
        action_type="create_ticket",
        parameters={"queue": "noc", "summary": "test"},
        expected_impact="passive",
        risk_level=0.0,
        requires_approval=False,
        utility_score=0.1,
    )
    result = ActionResult(
        action_id=action.id,
        success=True,
        pre_metrics={"link.latency_ms": 150.0},
        post_metrics={"link.latency_ms": 50.0},
        rollback_available=False,
        outcome="effective",
    )
    hypothesis = Hypothesis(
        root_cause="congestion",
        description="test",
        confidence=0.8,
        evidence=[],
    )
    learner.record_outcome(action, result, hypothesis)
    metrics = learner.get_metrics()
    assert metrics["total_actions"] >= 1
    assert 0.0 <= metrics["success_rate"] <= 1.0

    summary = learner.get_learning_summary()
    assert "total_actions" in summary
    assert "recommendations" in summary


# ---------------------------------------------------------------------------
# Test 9: Debate system produces DebateResult
# ---------------------------------------------------------------------------

def test_debate_system(state_dict):
    """DebateSystem.conduct_debate must produce a DebateResult with consensus."""
    from unittest.mock import patch, MagicMock
    from src.agents.debate import DebateSystem
    from src.models.schemas import ProposedAction, Hypothesis, DebateResult

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"stance": "support", "key_arguments": ["reduces congestion"], '
        '"risk_assessment": "low", "confidence": 0.8}'
    )
    mock_llm.invoke.return_value = mock_response

    action = ProposedAction(
        action_type="reroute",
        target_link="CR1-CR2",
        parameters={"from_link": "CR1-CR2", "to_link": "AGG1-CR1",
                    "traffic_fraction": 0.3, "source_node": "CR1",
                    "target_node": "CR2", "new_weight": 8.0},
        expected_impact="Reduce congestion on CR1-CR2.",
        risk_level=0.55,
        requires_approval=False,
        utility_score=0.6,
    )
    hypothesis = Hypothesis(
        root_cause="congestion",
        description="Link saturated",
        confidence=0.85,
        evidence=["utilization_pct=90"],
        affected_links=["CR1-CR2"],
    )

    with patch("src.agents.debate.ChatOpenAI", return_value=mock_llm):
        ds = DebateSystem()
        result = ds.conduct_debate(
            action=action,
            hypothesis=hypothesis,
            network_state=state_dict,
            causal_context={},
        )

    assert isinstance(result, DebateResult)
    assert result.final_decision in {"approve", "reject", "modify"}


# ---------------------------------------------------------------------------
# Test 10: Orchestrator single cycle without crashing
# ---------------------------------------------------------------------------

def test_orchestrator_single_cycle(orchestrator):
    """Orchestrator.run_loop(max_cycles=1) must complete without exception."""
    final_state = asyncio.get_event_loop().run_until_complete(
        orchestrator.run_loop(max_cycles=1)
    )
    assert isinstance(final_state, dict)
    orch_state = orchestrator.get_state()
    assert "is_running" in orch_state
    assert "cycle_count" in orch_state
    assert "pending_approvals" in orch_state
