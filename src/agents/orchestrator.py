from __future__ import annotations

import asyncio
import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.actor import ActorAgent
from src.agents.debate import DebateSystem
from src.agents.decider import DeciderAgent
from src.agents.learner import LearnerAgent
from src.agents.observer import ObserverAgent
from src.agents.reasoner import ReasonerAgent
from src.causal.causal_engine import CausalEngine
from src.models.schemas import (
    ActionResult,
    Anomaly,
    AuditEntry,
    Hypothesis,
    ProposedAction,
)
from src.safety.z3_verifier import Z3SafetyVerifier
from src.simulator.engine import SimulationEngine
from src.simulator.topology import NetworkTopology
from src.utils.logger import logger


class AgentState(TypedDict):
    network_state: dict[str, Any]
    anomalies: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    proposed_actions: list[dict[str, Any]]
    current_action: Optional[dict[str, Any]]
    debate_result: Optional[dict[str, Any]]
    verification_result: Optional[dict[str, Any]]
    action_result: Optional[dict[str, Any]]
    audit_log: Annotated[list[dict[str, Any]], operator.add]
    cycle_count: int
    decision_attempts: int
    should_continue: bool


class AgentOrchestrator:
    """Main LangGraph orchestrator for observe→reason→decide→act→learn loop."""

    def __init__(self, topology: NetworkTopology, simulation_engine: SimulationEngine) -> None:
        """Initialize all agent subsystems and compile workflow graph."""
        self.topology = topology
        self.engine = simulation_engine

        baselines = self._compute_initial_baselines()
        self.observer = ObserverAgent(topology, baselines=baselines)
        self.causal_engine = CausalEngine(topology)
        self.reasoner = ReasonerAgent(self.causal_engine, topology)
        self.z3_verifier = Z3SafetyVerifier(topology)
        self.decider = DeciderAgent(topology, self.causal_engine, self.z3_verifier)
        self.debate_system = DebateSystem()
        self.actor = ActorAgent(simulation_engine)
        self.learner = LearnerAgent()
        self.graph = self._build_graph()

        self.is_running = False
        self.pending_approvals: dict[str, ProposedAction] = {}
        self._latest_state: AgentState = self._initial_state()

    def _build_graph(self) -> Any:
        workflow: StateGraph[AgentState] = StateGraph(AgentState)

        workflow.add_node("observe", self._observe_node)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("decide", self._decide_node)
        workflow.add_node("debate", self._debate_node)
        workflow.add_node("verify", self._verify_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("learn", self._learn_node)

        workflow.set_entry_point("observe")

        # END is used as "no work this cycle" so run_loop controls outer repetition.
        workflow.add_conditional_edges(
            "observe",
            self._should_reason,
            {"reason": "reason", "observe": END},
        )
        workflow.add_edge("reason", "decide")
        workflow.add_conditional_edges(
            "decide",
            self._should_debate,
            {"debate": "debate", "verify": "verify"},
        )
        workflow.add_edge("debate", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._is_safe,
            {"act": "act", "decide": "decide", "stop": END},
        )
        workflow.add_edge("act", "learn")
        workflow.add_edge("learn", END)

        return workflow.compile()

    def _observe_node(self, state: AgentState) -> dict[str, Any]:
        telemetry = self.engine.get_current_state_dict()
        self.observer.ingest(telemetry)
        anomalies = self.observer.detect()
        return {
            "network_state": telemetry,
            "anomalies": [a.model_dump() for a in anomalies],
            "debate_result": None,
            "verification_result": None,
            "action_result": None,
            "decision_attempts": 0,
        }

    def _reason_node(self, state: AgentState) -> dict[str, Any]:
        anomalies = [Anomaly(**payload) for payload in state.get("anomalies", [])]
        hypotheses = self.reasoner.analyze(anomalies, state.get("network_state", {}))
        return {"hypotheses": [h.model_dump() for h in hypotheses]}

    def _decide_node(self, state: AgentState) -> dict[str, Any]:
        hypotheses = [Hypothesis(**payload) for payload in state.get("hypotheses", [])]
        network_state = state.get("network_state", {})
        actions = self.decider.evaluate(hypotheses, network_state)
        top_action = actions[0] if actions else None
        attempts = int(state.get("decision_attempts", 0)) + 1

        # RL suggestion — log alongside top action for comparison
        link_metrics: dict[str, dict[str, float]] = {}
        for link_id, metrics in network_state.get("links", {}).items():
            if isinstance(metrics, dict):
                link_metrics[str(link_id)] = {
                    k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
                }
        if link_metrics:
            rl_suggestion = self.decider.get_rl_suggestion(link_metrics)
            if rl_suggestion:
                logger.debug(
                    "RL suggestion: action={} target={} confidence={:.2f}",
                    rl_suggestion.get("action_type"),
                    rl_suggestion.get("target_link"),
                    float(rl_suggestion.get("confidence", 0.0)),
                )

        # Uncertainty-based escalation: if top action confidence is very low,
        # override to escalate rather than auto-act
        if top_action is not None:
            uncertainty_score = float(top_action.parameters.get("uncertainty_score", 0.0))
            if self.decider.should_escalate_for_uncertainty(uncertainty_score):
                logger.info(
                    "High forecast uncertainty ({:.3f}) — escalating action to require approval.",
                    uncertainty_score,
                )
                top_action = top_action.model_copy(update={"requires_approval": True})

        return {
            "proposed_actions": [a.model_dump() for a in actions],
            "current_action": top_action.model_dump() if top_action else None,
            "decision_attempts": attempts,
        }

    def _debate_node(self, state: AgentState) -> dict[str, Any]:
        current_action_data = state.get("current_action")
        hypotheses_data = state.get("hypotheses", [])
        if not current_action_data or not hypotheses_data:
            return {"debate_result": None}

        action = ProposedAction(**current_action_data)
        hypothesis = Hypothesis(**hypotheses_data[0])
        result = self.debate_system.conduct_debate(
            action=action,
            hypothesis=hypothesis,
            network_state=state.get("network_state", {}),
            causal_context={},
        )
        return {"debate_result": result.model_dump()}

    def _verify_node(self, state: AgentState) -> dict[str, Any]:
        current_action_data = state.get("current_action")
        if not current_action_data:
            return {"verification_result": {"is_safe": False, "reason": "No action selected"}}

        action = ProposedAction(**current_action_data)
        flat_state = self._flatten_network_state(state.get("network_state", {}))
        counterfactual = self.causal_engine.run_counterfactual(
            self.decider.to_counterfactual_payload(action),
            flat_state,
        )
        verification = self.z3_verifier.verify_action(
            action,
            flat_state,
            counterfactual.get("predicted_state", {}),
        )
        return {"verification_result": verification}

    def _act_node(self, state: AgentState) -> dict[str, Any]:
        current_action_data = state.get("current_action")
        if not current_action_data:
            return {"action_result": {"status": "no_action"}}

        action = ProposedAction(**current_action_data)
        if action.requires_approval:
            self.pending_approvals[action.id] = action
            return {"action_result": {"status": "pending_approval", "action_id": action.id}}

        result = self.actor.execute(action)
        return {"action_result": result.model_dump()}

    def _learn_node(self, state: AgentState) -> dict[str, Any]:
        action_result_data = state.get("action_result")
        current_action_data = state.get("current_action")
        hypotheses_data = state.get("hypotheses", [])

        if (
            isinstance(action_result_data, dict)
            and action_result_data.get("status") != "pending_approval"
            and current_action_data
            and hypotheses_data
        ):
            try:
                action = ProposedAction(**current_action_data)
                result = ActionResult(**action_result_data)
                hypothesis = Hypothesis(**hypotheses_data[0])
                self.learner.record_outcome(action, result, hypothesis)
            except Exception:
                pass

        anomaly_models: list[Anomaly] = []
        for payload in state.get("anomalies", []):
            try:
                anomaly_models.append(Anomaly(**payload))
            except Exception:
                continue

        hypothesis_model = None
        if hypotheses_data:
            try:
                hypothesis_model = Hypothesis(**hypotheses_data[0])
            except Exception:
                hypothesis_model = None

        proposed_action = None
        if current_action_data:
            try:
                proposed_action = ProposedAction(**current_action_data)
            except Exception:
                proposed_action = None

        outcome = None
        if isinstance(action_result_data, dict):
            maybe_outcome = action_result_data.get("outcome")
            if isinstance(maybe_outcome, str):
                outcome = maybe_outcome

        # Record anomaly detection samples for model retraining
        for anomaly in anomaly_models:
            entity_id = anomaly.node_id or anomaly.link_id or "unknown"
            self.learner.record_anomaly_detection_sample(
                entity_id=entity_id,
                metric=anomaly.metric_name,
                observed_value=float(anomaly.observed_value),
                expected_value=float(anomaly.expected_value),
                is_anomaly=True,
                scenario_type=hypothesis_model.root_cause if hypothesis_model else "unknown",
            )

        # Trigger retraining if enough samples accumulated
        anomaly_samples = len(self.learner._anomaly_detection_data)
        if anomaly_samples > 0 and anomaly_samples % 100 == 0:
            self.learner.trigger_model_retrain(
                reason=f"Accumulated {anomaly_samples} labeled anomaly samples",
                model_type="anomaly_detection",
            )

        # Periodically log learning summary
        cycle_num = int(state.get("cycle_count", 0)) + 1
        if cycle_num % 10 == 0:
            summary = self.learner.get_learning_summary()
            logger.info(
                "Learning summary cycle={}: actions={} success_rate={:.2f} llm_examples={}",
                cycle_num,
                summary.get("total_actions", 0),
                float(summary.get("success_rate", 0.0)),
                summary.get("llm_examples", 0),
            )

        audit_entry = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            decision_id=(proposed_action.id if proposed_action else "none"),
            phase="learn",
            anomalies=anomaly_models,
            hypothesis=hypothesis_model,
            proposed_action=proposed_action,
            rationale="Cycle complete",
            outcome=outcome,
        )
        return {
            "audit_log": [audit_entry.model_dump()],
            "cycle_count": cycle_num,
        }

    def _should_reason(self, state: AgentState) -> str:
        return "reason" if state.get("anomalies") else "observe"

    def _should_debate(self, state: AgentState) -> str:
        action_data = state.get("current_action")
        if action_data:
            action = ProposedAction(**action_data)
            if self.decider.should_trigger_debate(action):
                return "debate"
        return "verify"

    def _is_safe(self, state: AgentState) -> str:
        result = state.get("verification_result") or {}
        if bool(result.get("is_safe", False)):
            return "act"
        if int(state.get("decision_attempts", 0)) >= 3:
            return "stop"
        return "decide"

    async def run_loop(self, max_cycles: int | None = None) -> AgentState:
        """Run orchestration loop for `max_cycles` (or until stopped)."""
        self.is_running = True
        state = self._latest_state
        cycle = 0

        while self.is_running and (max_cycles is None or cycle < max_cycles):
            state = self.graph.invoke(state)
            state["cycle_count"] = int(state.get("cycle_count", cycle))
            self._latest_state = state
            cycle += 1
            await asyncio.sleep(1)

        return state

    def stop(self) -> None:
        """Stop orchestration loop."""
        self.is_running = False

    def approve_action(self, action_id: str) -> bool:
        """Approve a pending action and execute immediately."""
        if action_id not in self.pending_approvals:
            return False
        action = self.pending_approvals.pop(action_id)
        action.requires_approval = False
        self.actor.execute(action)
        return True

    def get_state(self) -> dict[str, Any]:
        """Return current high-level orchestrator state summary."""
        return {
            "is_running": self.is_running,
            "cycle_count": self._latest_state.get("cycle_count", 0),
            "pending_approvals": list(self.pending_approvals.keys()),
            "active_anomalies": [a.model_dump() for a in self.observer.get_active_anomalies()],
            "last_action_result": self._latest_state.get("action_result"),
        }

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return accumulated audit log entries."""
        return list(self._latest_state.get("audit_log", []))

    def _compute_initial_baselines(self) -> dict[str, dict[str, float]]:
        snapshot = self.engine.get_current_state_dict()
        baselines: dict[str, dict[str, float]] = {}
        for node_id, metrics in snapshot.get("nodes", {}).items():
            if isinstance(metrics, dict):
                baselines[str(node_id)] = {
                    str(metric): float(value)
                    for metric, value in metrics.items()
                    if isinstance(value, (int, float))
                }
        for link_id, metrics in snapshot.get("links", {}).items():
            if isinstance(metrics, dict):
                baselines[str(link_id)] = {
                    str(metric): float(value)
                    for metric, value in metrics.items()
                    if isinstance(value, (int, float))
                }
        return baselines

    def _flatten_network_state(self, network_state: dict[str, Any]) -> dict[str, dict[str, float]]:
        if "nodes" not in network_state and "links" not in network_state:
            return {
                str(entity): {
                    str(metric): float(value)
                    for metric, value in metrics.items()
                    if isinstance(value, (int, float))
                }
                for entity, metrics in network_state.items()
                if isinstance(metrics, dict)
            }
        flat: dict[str, dict[str, float]] = {}
        for node_id, metrics in network_state.get("nodes", {}).items():
            if isinstance(metrics, dict):
                flat[str(node_id)] = {
                    str(metric): float(value)
                    for metric, value in metrics.items()
                    if isinstance(value, (int, float))
                }
        for link_id, metrics in network_state.get("links", {}).items():
            if isinstance(metrics, dict):
                flat[str(link_id)] = {
                    str(metric): float(value)
                    for metric, value in metrics.items()
                    if isinstance(value, (int, float))
                }
        return flat

    def _initial_state(self) -> AgentState:
        return {
            "network_state": {},
            "anomalies": [],
            "hypotheses": [],
            "proposed_actions": [],
            "current_action": None,
            "debate_result": None,
            "verification_result": None,
            "action_result": None,
            "audit_log": [],
            "cycle_count": 0,
            "decision_attempts": 0,
            "should_continue": True,
        }
