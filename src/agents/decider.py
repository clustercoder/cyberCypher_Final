from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.causal.causal_engine import CausalEngine
from src.models.schemas import AuditEntry, Hypothesis, ProposedAction
from src.safety.z3_verifier import Z3SafetyVerifier
from src.simulator.topology import NetworkTopology
from src.utils.logger import logger

try:
    from src.models.rl_traffic_engineering import RLTrafficEngineer
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False

try:
    from src.simulator.digital_twin import DigitalTwin
    _TWIN_AVAILABLE = True
except ImportError:
    _TWIN_AVAILABLE = False

_SLA_PENALTY = {
    "critical": 1.0,
    "high": 0.7,
    "medium": 0.4,
    "low": 0.1,
}

_ACTION_COST = {
    "reroute": 0.3,
    "rate_limit": 0.2,
    "config_rollback": 0.4,
    "escalate": 0.1,
    "create_ticket": 0.05,
    "scale_capacity": 0.5,
}

_OPERATIONAL_RISK = {
    "reroute": 0.4,
    "rate_limit": 0.2,
    "config_rollback": 0.5,
    "escalate": 0.0,
    "create_ticket": 0.0,
    "scale_capacity": 0.3,
}

_BASE_RISK = {
    "reroute": 0.55,
    "rate_limit": 0.35,
    "config_rollback": 0.6,
    "escalate": 0.15,
    "create_ticket": 0.05,
    "scale_capacity": 0.5,
}


class DeciderAgent:
    """Evaluates hypotheses and selects the safest highest-utility intervention."""

    def __init__(
        self,
        topology: NetworkTopology,
        causal_engine: CausalEngine,
        z3_verifier: Z3SafetyVerifier,
    ) -> None:
        """Initialize decision agent with topology, causal simulation, and safety verifier."""
        self.topology = topology
        self.causal_engine = causal_engine
        self.z3_verifier = z3_verifier

        self._action_effectiveness: dict[str, float] = {
            "reroute": 0.7,
            "rate_limit": 0.68,
            "config_rollback": 0.72,
            "escalate": 0.45,
            "create_ticket": 0.25,
            "scale_capacity": 0.75,
        }
        self._action_confidence: dict[str, float] = {}

        # Optional RL traffic engineer
        self.rl_engineer: RLTrafficEngineer | None = None  # type: ignore[type-arg]
        if _RL_AVAILABLE:
            try:
                link_ids = [f"{u}-{v}" for u, v in topology._graph.edges()]
                self.rl_engineer = RLTrafficEngineer(link_ids=link_ids)  # type: ignore[assignment]
                logger.info("RLTrafficEngineer initialized.")
            except Exception as exc:
                logger.warning("RLTrafficEngineer init failed: {}", exc)

        # Optional digital twin pre-validator
        self.digital_twin: DigitalTwin | None = None  # type: ignore[type-arg]
        if _TWIN_AVAILABLE:
            try:
                self.digital_twin = DigitalTwin(topology=topology)  # type: ignore[assignment]
                logger.info("DigitalTwin initialized.")
            except Exception as exc:
                logger.warning("DigitalTwin init failed: {}", exc)

    def evaluate(
        self, hypotheses: list[Hypothesis], network_state: dict[str, Any]
    ) -> list[ProposedAction]:
        """Generate, score, and safety-verify candidate actions."""
        if not hypotheses:
            return []

        flat_state = self._flatten_network_state(network_state)
        scored_pairs: list[tuple[ProposedAction, Hypothesis]] = []

        for hypothesis in hypotheses:
            candidates = self._generate_candidate_actions(hypothesis, network_state)
            for action in candidates:
                utility = self._compute_utility(action, hypothesis, network_state)
                updated = action.model_copy(update={"utility_score": utility})
                scored_pairs.append((updated, hypothesis))
                self._action_confidence[updated.id] = hypothesis.confidence

        scored_pairs.sort(key=lambda pair: pair[0].utility_score, reverse=True)
        if not scored_pairs:
            return []

        decided: list[ProposedAction] = []
        safe_choice_found = False

        for action, hypothesis in scored_pairs:
            if safe_choice_found:
                decided.append(action)
                continue

            counterfactual_payload = self.to_counterfactual_payload(action)
            counterfactual = self.causal_engine.run_counterfactual(counterfactual_payload, flat_state)
            predicted_state = counterfactual.get("predicted_state", {})

            # Digital twin pre-validation — refines risk_level before Z3 check
            twin_risk = action.risk_level
            if self.digital_twin is not None:
                try:
                    twin_result = self.digital_twin.simulate_action(action, flat_state)
                    twin_summary = twin_result.get("summary", {})
                    twin_risk = float(twin_summary.get("risk_score", action.risk_level))
                    if not twin_summary.get("safe", True):
                        twin_risk = max(twin_risk, 0.75)
                    logger.debug(
                        "DigitalTwin: action={} twin_risk={:.3f} safe={}",
                        action.action_type,
                        twin_risk,
                        twin_summary.get("safe"),
                    )
                except Exception as exc:
                    logger.warning("DigitalTwin simulation failed: {}", exc)

            # record=False: pre-check only; rate-limit history is committed by
            # the orchestrator's _verify_node when the action is actually executed.
            verification = self.z3_verifier.verify_action(action, flat_state, predicted_state, record=False)
            is_safe = bool(verification.get("is_safe", False))

            combined_risk = float(
                min(1.0, max(action.risk_level, twin_risk, float(counterfactual.get("risk_score", 0.0))))
            )
            verified_action = action.model_copy(
                update={
                    "z3_verified": is_safe,
                    "z3_proof": str(verification.get("proof", "")),
                    "risk_level": combined_risk,
                }
            )

            if not is_safe:
                decided.append(verified_action)
                continue

            final_action = self._apply_decision_policy(verified_action, hypothesis.confidence)
            decided.append(final_action)
            safe_choice_found = True

        decided.sort(key=lambda a: a.utility_score, reverse=True)
        return decided

    def _generate_candidate_actions(
        self, hypothesis: Hypothesis, network_state: dict[str, Any]
    ) -> list[ProposedAction]:
        """Generate action candidates according to hypothesis type."""
        hypothesis_type = self._classify_hypothesis(hypothesis)
        action_map = {
            "congestion": ["reroute", "rate_limit"],
            "hardware": ["escalate", "create_ticket"],
            "ddos": ["rate_limit", "escalate"],
            "misconfiguration": ["config_rollback", "create_ticket"],
            "link_down": ["reroute", "escalate"],
        }
        action_types = action_map.get(hypothesis_type, ["create_ticket", "escalate"])

        target_node, target_link = self._extract_targets(hypothesis)
        actions: list[ProposedAction] = []

        for action_type in action_types:
            params = self._build_action_parameters(
                action_type=action_type,
                hypothesis=hypothesis,
                network_state=network_state,
                target_node=target_node,
                target_link=target_link,
            )
            expected_impact = self._build_expected_impact(action_type, target_node, target_link)
            severity = self._infer_severity(hypothesis)
            severity_boost = _SLA_PENALTY[severity] * 0.25
            risk = min(1.0, _BASE_RISK.get(action_type, 0.4) + (1.0 - hypothesis.confidence) * 0.2 + severity_boost)

            # For link-targeting actions ensure we have a valid target_link
            effective_link = target_link
            if action_type in {"reroute", "rate_limit", "scale_capacity"} and effective_link is None:
                effective_link = hypothesis.affected_links[0] if hypothesis.affected_links else None

            action = ProposedAction(
                action_type=action_type,  # type: ignore[arg-type]
                target_node=target_node if action_type not in {"reroute", "rate_limit", "scale_capacity"} else None,
                target_link=effective_link if action_type in {"reroute", "rate_limit", "scale_capacity"} else None,
                parameters=params,
                expected_impact=expected_impact,
                risk_level=float(risk),
                requires_approval=False,
                utility_score=0.0,
                z3_verified=False,
            )
            actions.append(action)

        return actions

    def _compute_utility(
        self, action: ProposedAction, hypothesis: Hypothesis, state: dict[str, Any]
    ) -> float:
        """Score utility using impact, risk/cost, and probability of remediation success."""
        expected_customer_impact = self._estimate_customer_impact(hypothesis)
        severity = self._infer_severity(hypothesis)
        sla_penalty = _SLA_PENALTY[severity]

        action_cost = _ACTION_COST[action.action_type]
        operational_risk = _OPERATIONAL_RISK[action.action_type]
        action_effectiveness = self._action_effectiveness.get(action.action_type, 0.7)
        probability_of_fix = hypothesis.confidence * action_effectiveness
        benefit = sla_penalty

        utility = (
            -(expected_customer_impact * sla_penalty)
            - (action_cost * operational_risk)
            + (probability_of_fix * benefit)
        )
        return round(float(utility), 6)

    def get_rl_suggestion(
        self, link_metrics: dict[str, dict[str, float]]
    ) -> dict[str, Any] | None:
        """Return RL policy action suggestion for current link metrics, or None if unavailable."""
        if self.rl_engineer is None:
            return None
        try:
            return self.rl_engineer.suggest_action(link_metrics)
        except Exception as exc:
            logger.warning("RL suggestion failed: {}", exc)
            return None

    def should_escalate_for_uncertainty(self, uncertainty_score: float) -> bool:
        """Return True if forecast uncertainty is high enough to require human review."""
        return uncertainty_score > 0.3

    def should_trigger_debate(self, action: ProposedAction) -> bool:
        """Return True for high-risk actions or low-confidence disruptive operations."""
        confidence = self._action_confidence.get(action.id, 1.0)
        if action.risk_level >= 0.7:
            return True
        if action.action_type in {"reroute", "config_rollback"} and confidence < 0.85:
            return True
        return False

    def get_decision_rationale(self, action: ProposedAction, hypothesis: Hypothesis) -> AuditEntry:
        """Build structured decision audit entry."""
        rationale = (
            f"Selected action '{action.action_type}' for root cause '{hypothesis.root_cause}' "
            f"with utility={action.utility_score:.3f}, risk={action.risk_level:.2f}, "
            f"z3_verified={action.z3_verified}."
        )
        return AuditEntry(
            timestamp=datetime.now(timezone.utc),
            decision_id=action.id,
            phase="decide",
            anomalies=[],
            hypothesis=hypothesis,
            proposed_action=action,
            rationale=rationale,
        )

    def to_counterfactual_payload(self, action: ProposedAction) -> dict[str, Any]:
        """Map ProposedAction to CausalEngine counterfactual action payload."""
        payload = {"action_type": action.action_type}
        params = dict(action.parameters)

        if action.action_type == "reroute":
            payload.update(
                {
                    "from_link": params.get("from_link") or action.target_link,
                    "to_link": params.get("to_link"),
                    "traffic_fraction": float(params.get("traffic_fraction", 0.3)),
                }
            )
        elif action.action_type == "rate_limit":
            payload.update(
                {
                    "target_link": params.get("target_link") or action.target_link,
                    "reduction_pct": float(params.get("reduction_pct", 20.0)),
                }
            )
        return payload

    def _apply_decision_policy(self, action: ProposedAction, confidence: float) -> ProposedAction:
        if confidence < 0.6:
            return ProposedAction(
                action_type="create_ticket",
                target_node=action.target_node,
                target_link=action.target_link,
                parameters={
                    "reason": "low_confidence",
                    "original_action_type": action.action_type,
                },
                expected_impact=(
                    f"Passive handling due to low confidence ({confidence:.2f}) for {action.action_type}."
                ),
                risk_level=0.0,
                requires_approval=False,
                utility_score=action.utility_score,
                z3_verified=action.z3_verified,
                z3_proof=action.z3_proof,
            )

        requires_approval = False
        if action.risk_level >= 0.7:
            requires_approval = True
        elif action.risk_level < 0.4:
            requires_approval = False
        elif confidence < 0.85:
            requires_approval = True

        return action.model_copy(update={"requires_approval": requires_approval})

    def _classify_hypothesis(self, hypothesis: Hypothesis) -> str:
        text = " ".join(
            [
                hypothesis.description,
                hypothesis.root_cause,
                *hypothesis.evidence,
                " ".join(hypothesis.recommended_actions),
            ]
        ).lower()

        if any(token in text for token in ("ddos", "surge", "flood", "attack")):
            return "ddos"
        if any(token in text for token in ("misconfig", "configuration", "rollback", "route leak")):
            return "misconfiguration"
        if any(token in text for token in ("link down", "fiber cut", "down", "disconnected")):
            return "link_down"
        if any(token in text for token in ("hardware", "temperature", "cpu", "memory", "buffer")):
            return "hardware"
        if any(token in text for token in ("utilization", "congestion", "latency", "packet_loss")):
            return "congestion"
        return "congestion"

    def _extract_targets(self, hypothesis: Hypothesis) -> tuple[str | None, str | None]:
        root = hypothesis.root_cause
        entity = root.rsplit("_", 1)[0] if "_" in root else root
        if "-" in entity:
            return None, entity
        if entity:
            return entity, None
        if hypothesis.affected_links:
            return None, hypothesis.affected_links[0]
        if hypothesis.affected_nodes:
            return hypothesis.affected_nodes[0], None
        return None, None

    def _build_action_parameters(
        self,
        action_type: str,
        hypothesis: Hypothesis,
        network_state: dict[str, Any],
        target_node: str | None,
        target_link: str | None,
    ) -> dict[str, Any]:
        if action_type == "reroute":
            from_link = target_link or (hypothesis.affected_links[0] if hypothesis.affected_links else None)
            alt_link = self._find_alternate_link(from_link)
            source_node, target = self._parse_link_endpoints(from_link)
            return {
                "from_link": from_link,
                "to_link": alt_link,
                "traffic_fraction": 0.3,
                "source_node": source_node,
                "target_node": target,
                "new_weight": 8.0,
            }

        if action_type == "rate_limit":
            link = target_link or (hypothesis.affected_links[0] if hypothesis.affected_links else None)
            return {
                "target_link": link,
                "limit_pct": 70.0,
                "reduction_pct": 20.0,
            }

        if action_type == "scale_capacity":
            link = target_link or (hypothesis.affected_links[0] if hypothesis.affected_links else None)
            current_capacity = self._lookup_link_capacity(link)
            return {
                "target_link": link,
                "new_capacity_gbps": round(current_capacity * 1.2, 4),
            }

        if action_type == "config_rollback":
            return {"rollback_token": "latest"}

        if action_type == "escalate":
            return {"priority": "high", "team": "netops"}

        if action_type == "create_ticket":
            return {"queue": "noc", "summary": hypothesis.description}

        return {}

    def _build_expected_impact(
        self, action_type: str, target_node: str | None, target_link: str | None
    ) -> str:
        target = target_link or target_node or "network segment"
        if action_type == "reroute":
            return f"Shift traffic away from {target} to relieve congestion and latency."
        if action_type == "rate_limit":
            return f"Throttle suspicious or excessive flow on {target} to protect stability."
        if action_type == "scale_capacity":
            return f"Increase available throughput on {target} to absorb burst demand."
        if action_type == "config_rollback":
            return f"Revert recent config changes affecting {target}."
        if action_type == "escalate":
            return f"Escalate incident impacting {target} to on-call experts."
        return f"Create operator ticket for manual investigation on {target}."

    def _infer_severity(self, hypothesis: Hypothesis) -> str:
        text = " ".join([hypothesis.description, *hypothesis.evidence]).lower()
        if "critical" in text:
            return "critical"
        if "high" in text:
            return "high"
        if "low" in text:
            return "low"
        return "medium"

    def _estimate_customer_impact(self, hypothesis: Hypothesis) -> float:
        all_nodes = {node["node_id"]: node for node in self.topology.get_all_nodes()}
        total_customers = sum(int(node.get("customers", 0)) for node in all_nodes.values())
        if total_customers <= 0:
            return 0.0

        affected_nodes = set(hypothesis.affected_nodes)
        for link_id in hypothesis.affected_links:
            src, dst = self._parse_link_endpoints(link_id)
            if src:
                affected_nodes.add(src)
            if dst:
                affected_nodes.add(dst)

        impacted_customers = sum(
            int(all_nodes[node_id].get("customers", 0))
            for node_id in affected_nodes
            if node_id in all_nodes
        )
        return float(min(1.0, impacted_customers / total_customers))

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

    def _find_alternate_link(self, from_link: str | None) -> str | None:
        if not from_link:
            return None
        src, dst = self._parse_link_endpoints(from_link)
        if not src or not dst:
            return None
        for neighbor in self.topology.get_neighbors(src):
            candidate = self.topology.get_link_id(src, neighbor)
            if candidate != from_link:
                return candidate
        return None

    def _parse_link_endpoints(self, link_id: str | None) -> tuple[str | None, str | None]:
        if not link_id or "-" not in link_id:
            return None, None
        parts = link_id.split("-")
        if len(parts) < 2:
            return None, None
        return parts[0], parts[1]

    def _lookup_link_capacity(self, link_id: str | None) -> float:
        if link_id is None:
            return 100.0
        for link in self.topology.get_all_links():
            if link["link_id"] == link_id:
                return float(link.get("capacity_gbps", 100.0))
        return 100.0
