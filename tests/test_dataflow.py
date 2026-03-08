"""Data flow simulation tests — 8 pipeline stages.

Verifies that data transforms correctly through each stage of the
observe→reason→decide→act→learn pipeline.
"""
from __future__ import annotations

import os
import json
from datetime import datetime, timezone

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-dataflow")


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


# ---------------------------------------------------------------------------
# Stage 1: Telemetry dict structure
# ---------------------------------------------------------------------------

def test_stage1_telemetry_structure(engine):
    """Telemetry dict must have nodes and links with numeric metrics."""
    state = engine.get_current_state_dict()
    assert isinstance(state, dict)
    assert "nodes" in state and "links" in state

    for node_id, metrics in state["nodes"].items():
        assert isinstance(node_id, str)
        assert isinstance(metrics, dict)
        for k, v in metrics.items():
            assert isinstance(v, (int, float, str, bool)), (
                f"Node {node_id} metric {k} has unexpected type {type(v)}"
            )

    for link_id, metrics in state["links"].items():
        assert isinstance(link_id, str)
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# Stage 2: Anomaly schema after observer ingestion
# ---------------------------------------------------------------------------

def test_stage2_anomaly_schema(topology, engine):
    """Observer anomalies must be valid Anomaly pydantic models."""
    from src.agents.observer import ObserverAgent
    from src.models.schemas import Anomaly

    state = engine.get_current_state_dict()
    # spike all nodes
    state["nodes"] = {
        k: {**v, "cpu_pct": 99.0, "memory_pct": 99.0}
        for k, v in state.get("nodes", {}).items()
    }
    observer = ObserverAgent(topology, baselines={})
    observer.ingest(state)
    anomalies = observer.detect()

    assert isinstance(anomalies, list)
    for a in anomalies:
        assert isinstance(a, Anomaly)
        assert isinstance(a.node_id, str) or a.node_id is None
        assert 0.0 <= a.confidence <= 1.0
        assert a.severity in {"low", "medium", "high", "critical"}
        # Must be JSON-serialisable
        json.dumps(a.model_dump(), default=str)


# ---------------------------------------------------------------------------
# Stage 3: Hypothesis schema from Reasoner (mocked LLM)
# ---------------------------------------------------------------------------

def test_stage3_hypothesis_schema(topology, causal_engine, engine):
    """Hypotheses from ReasonerAgent must be valid Hypothesis pydantic models."""
    from unittest.mock import patch, MagicMock
    from src.agents.reasoner import ReasonerAgent
    from src.models.schemas import Anomaly, Hypothesis

    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = json.dumps({
        "root_cause": "link_congestion",
        "description": "Utilization saturated on upstream link",
        "confidence": 0.88,
        "evidence": ["utilization_pct=92"],
        "affected_nodes": ["CR1"],
        "affected_links": ["CR1-CR2"],
        "recommended_actions": ["reroute"],
    })
    mock_llm.invoke.return_value = mock_resp

    with patch("src.agents.reasoner.ChatOpenAI", return_value=mock_llm):
        reasoner = ReasonerAgent(causal_engine, topology)

    state = engine.get_current_state_dict()
    anomalies = [
        Anomaly(
            node_id="CR1",
            metric_name="utilization_pct",
            observed_value=92.0,
            expected_value=60.0,
            severity="high",
            confidence=0.88,
            detector_type="threshold",
            timestamp=datetime.now(timezone.utc),
        )
    ]
    hypotheses = reasoner.analyze(anomalies, state)
    assert isinstance(hypotheses, list)
    for h in hypotheses:
        assert isinstance(h, Hypothesis)
        assert isinstance(h.root_cause, str)
        assert 0.0 <= h.confidence <= 1.0
        json.dumps(h.model_dump(), default=str)


# ---------------------------------------------------------------------------
# Stage 4: ProposedAction schema from Decider
# ---------------------------------------------------------------------------

def test_stage4_proposed_action_schema(decider, engine):
    """Actions from DeciderAgent must be valid ProposedAction models."""
    from src.models.schemas import Hypothesis, ProposedAction

    state = engine.get_current_state_dict()
    hypotheses = [
        Hypothesis(
            root_cause="congestion",
            description="Link CR1-CR2 saturated",
            confidence=0.87,
            evidence=["utilization_pct=90"],
            affected_links=["CR1-CR2"],
        )
    ]
    actions = decider.evaluate(hypotheses, state)
    assert isinstance(actions, list)
    assert len(actions) > 0

    for a in actions:
        assert isinstance(a, ProposedAction)
        assert isinstance(a.action_type, str)
        assert 0.0 <= a.risk_level <= 1.0
        assert 0.0 <= a.utility_score <= 1.0
        json.dumps(a.model_dump(), default=str)

    # Sorted descending by utility
    if len(actions) > 1:
        assert actions[0].utility_score >= actions[1].utility_score


# ---------------------------------------------------------------------------
# Stage 5: Z3 verification verdict structure
# ---------------------------------------------------------------------------

def test_stage5_z3_verdict_structure(z3_verifier, decider, engine):
    """Z3 verdict dict must have is_safe bool and reason string."""
    from src.models.schemas import Hypothesis

    state = engine.get_current_state_dict()
    hypotheses = [
        Hypothesis(
            root_cause="congestion",
            description="Saturated link",
            confidence=0.9,
            evidence=["utilization_pct=91"],
            affected_links=["CR1-CR2"],
        )
    ]
    actions = decider.evaluate(hypotheses, state)
    if not actions:
        pytest.skip("No actions generated")

    flat = {
        k: {mk: float(mv) for mk, mv in v.items() if isinstance(mv, (int, float))}
        for k, v in state.get("nodes", {}).items()
        if isinstance(v, dict)
    }
    verdict = z3_verifier.verify_action(actions[0], flat, {})
    assert isinstance(verdict, dict)
    assert "is_safe" in verdict
    assert isinstance(verdict["is_safe"], bool)
    assert "proof" in verdict or "details" in verdict or "violations" in verdict


# ---------------------------------------------------------------------------
# Stage 6: ActionResult schema from Actor
# ---------------------------------------------------------------------------

def test_stage6_action_result_schema(engine):
    """Actor.execute must produce a valid ActionResult with expected fields."""
    from src.agents.actor import ActorAgent
    from src.models.schemas import ProposedAction, ActionResult

    actor = ActorAgent(engine)
    action = ProposedAction(
        action_type="create_ticket",
        parameters={"queue": "noc", "summary": "dataflow test"},
        expected_impact="Creates NOC ticket for manual review.",
        risk_level=0.0,
        requires_approval=False,
        utility_score=0.1,
    )
    result = actor.execute(action)
    assert isinstance(result, ActionResult)
    assert isinstance(result.success, bool)
    assert result.outcome in {"effective", "partially_effective", "ineffective", "harmful"}
    assert isinstance(result.rollback_available, bool)
    json.dumps(result.model_dump(), default=str)


# ---------------------------------------------------------------------------
# Stage 7: Learner metrics accumulation
# ---------------------------------------------------------------------------

def test_stage7_learner_metrics(engine):
    """Learner metrics must accumulate correctly across multiple outcomes."""
    from src.agents.learner import LearnerAgent
    from src.models.schemas import ProposedAction, ActionResult, Hypothesis

    learner = LearnerAgent()
    hypothesis = Hypothesis(
        root_cause="congestion",
        description="test",
        confidence=0.8,
        evidence=[],
    )

    outcomes = [
        ("effective", True),
        ("effective", True),
        ("ineffective", False),
        ("partially_effective", True),
    ]
    for outcome_str, success_bool in outcomes:
        action = ProposedAction(
            action_type="create_ticket",
            parameters={"queue": "noc", "summary": f"test {outcome_str}"},
            expected_impact="passive",
            risk_level=0.0,
            requires_approval=False,
            utility_score=0.1,
        )
        result = ActionResult(
            action_id=action.id,
            success=success_bool,
            pre_metrics={},
            post_metrics={},
            rollback_available=False,
            outcome=outcome_str,
        )
        learner.record_outcome(action, result, hypothesis)

    metrics = learner.get_metrics()
    assert metrics["total_actions"] == 4
    assert 0.0 <= metrics["success_rate"] <= 1.0
    # success_rate is derived from pre/post metrics by _label_outcome, not ActionResult.outcome

    summary = learner.get_learning_summary()
    assert "total_actions" in summary
    assert "success_rate" in summary
    assert "recommendations" in summary


# ---------------------------------------------------------------------------
# Stage 8: End-to-end JSON serialisability of full pipeline output
# ---------------------------------------------------------------------------

def test_stage8_full_pipeline_json_serialisable(topology, engine):
    """All pipeline outputs must be JSON serialisable end-to-end."""
    from unittest.mock import patch, MagicMock
    from src.agents.observer import ObserverAgent
    from src.agents.reasoner import ReasonerAgent
    from src.agents.actor import ActorAgent
    from src.causal.causal_engine import CausalEngine
    from src.safety.z3_verifier import Z3SafetyVerifier
    from src.agents.decider import DeciderAgent
    from src.models.schemas import Anomaly, Hypothesis

    # Observer
    observer = ObserverAgent(topology, baselines={})
    state = engine.get_current_state_dict()
    state["nodes"] = {
        k: {**v, "cpu_pct": 95.0}
        for k, v in state.get("nodes", {}).items()
    }
    observer.ingest(state)
    anomalies = observer.detect()

    # Reasoner (mocked)
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = json.dumps({
        "root_cause": "cpu_overload",
        "description": "CPU saturated",
        "confidence": 0.82,
        "evidence": ["cpu_pct=95"],
        "affected_nodes": ["CR1"],
        "affected_links": [],
        "recommended_actions": ["create_ticket"],
    })
    mock_llm.invoke.return_value = mock_resp

    causal_engine = CausalEngine(topology)
    with patch("src.agents.reasoner.ChatOpenAI", return_value=mock_llm):
        reasoner = ReasonerAgent(causal_engine, topology)

    if anomalies:
        hypotheses = reasoner.analyze(anomalies[:1], state)
    else:
        hypotheses = [
            Hypothesis(
                root_cause="cpu_overload",
                description="CPU saturated",
                confidence=0.82,
                evidence=["cpu_pct=95"],
            )
        ]

    # Decider
    z3 = Z3SafetyVerifier(topology)
    decider = DeciderAgent(topology, causal_engine, z3)
    actions = decider.evaluate(hypotheses, state)

    # Actor
    actor = ActorAgent(engine)
    if actions:
        result = actor.execute(actions[0])
        result_dict = result.model_dump()
        json.dumps(result_dict, default=str)

    # Full pipeline dicts must be serialisable
    pipeline_output = {
        "state_keys": list(state.keys()),
        "anomaly_count": len(anomalies),
        "hypothesis_count": len(hypotheses),
        "action_count": len(actions),
        "anomalies": [a.model_dump() for a in anomalies],
        "hypotheses": [h.model_dump() for h in hypotheses],
        "actions": [a.model_dump() for a in actions],
    }
    serialised = json.dumps(pipeline_output, default=str)
    assert len(serialised) > 10
