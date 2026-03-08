"""Digital twin for pre-validation of network actions.

Provides fast-forward simulation of proposed actions against a copy of the
current network state, without touching the live simulation engine.
"""
from __future__ import annotations

import copy
from typing import Any

from src.models.schemas import ProposedAction
from src.simulator.topology import NetworkTopology

# Risk weights: utilization (40%), latency (30%), packet_loss (20%), sla (10%)
_RISK_WEIGHTS = {"utilization_pct": 0.4, "latency_ms": 0.3, "packet_loss_pct": 0.2, "sla": 0.1}
_SAFE_UTIL_LIMIT = 85.0
_SIMULATION_STEPS = 5


class DigitalTwin:
    """Lightweight digital twin for action pre-validation.

    Clones the current network state, applies a proposed action, and runs a
    configurable number of fast-forward steps to predict the post-action
    network state before committing to the live environment.

    Parameters
    ----------
    topology:
        NetworkTopology used for link/node relationships.
    """

    def __init__(self, topology: NetworkTopology) -> None:
        self.topology = topology

    def simulate_action(
        self,
        action: ProposedAction,
        current_state: dict[str, dict[str, float]],
        steps: int = _SIMULATION_STEPS,
    ) -> dict[str, Any]:
        """Simulate the proposed action and return predictions + safety summary.

        Parameters
        ----------
        action:
            The ProposedAction to pre-validate.
        current_state:
            Flat {entity_id: {metric: value}} network state snapshot.
        steps:
            Number of fast-forward simulation steps to run.

        Returns
        -------
        dict with keys:
        - ``predicted_states``: list[dict] of length ``steps`` (state per step)
        - ``summary``: dict with max_utilization, risk_score, safe (bool), rationale
        """
        state = copy.deepcopy(current_state)
        state = self._apply_action_to_state(action, state)

        predicted_states: list[dict[str, dict[str, float]]] = []
        for _ in range(steps):
            state = self._advance_state(state)
            predicted_states.append(copy.deepcopy(state))

        summary = self._compute_summary(action, predicted_states)
        return {"predicted_states": predicted_states, "summary": summary}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action_to_state(
        self,
        action: ProposedAction,
        state: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Apply the action's expected effect to the cloned state."""
        action_type = action.action_type
        params = dict(action.parameters)

        if action_type == "reroute":
            from_link = params.get("from_link") or action.target_link
            to_link = params.get("to_link")
            traffic_fraction = float(params.get("traffic_fraction", 0.3))

            if from_link and from_link in state:
                m = state[from_link]
                current_util = float(m.get("utilization_pct", 50.0))
                transferred = current_util * traffic_fraction
                m["utilization_pct"] = max(0.0, current_util - transferred)
                m["latency_ms"] = float(m.get("latency_ms", 5.0)) * 0.9

            if to_link and to_link in state:
                m = state[to_link]
                current_util = float(m.get("utilization_pct", 30.0))
                # assume from_link util was ~50% if not known
                from_util = float(state.get(from_link or "", {}).get("utilization_pct", 50.0)) if from_link else 50.0
                transferred = from_util * traffic_fraction
                m["utilization_pct"] = min(100.0, current_util + transferred * 0.7)

        elif action_type == "rate_limit":
            target_link = params.get("target_link") or action.target_link
            reduction_pct = float(params.get("reduction_pct", 20.0))
            if target_link and target_link in state:
                m = state[target_link]
                current_util = float(m.get("utilization_pct", 50.0))
                m["utilization_pct"] = max(0.0, current_util * (1.0 - reduction_pct / 100.0))
                m["packet_loss_pct"] = max(0.0, float(m.get("packet_loss_pct", 0.0)) * 0.7)

        elif action_type == "config_rollback":
            # Optimistic: assume rollback halves congestion and latency anomalies
            for entity_metrics in state.values():
                if isinstance(entity_metrics, dict):
                    if float(entity_metrics.get("utilization_pct", 0.0)) > 70.0:
                        entity_metrics["utilization_pct"] *= 0.8
                    if float(entity_metrics.get("latency_ms", 0.0)) > 20.0:
                        entity_metrics["latency_ms"] *= 0.85

        elif action_type == "scale_capacity":
            target_link = params.get("target_link") or action.target_link
            if target_link and target_link in state:
                m = state[target_link]
                current_cap = float(params.get("new_capacity_gbps", 120.0))
                # If capacity increases by 20%, effective utilization drops proportionally
                old_cap = current_cap / 1.2
                m["utilization_pct"] = float(m.get("utilization_pct", 50.0)) * (old_cap / current_cap)

        return state

    def _advance_state(
        self, state: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Simulate one step of state evolution (mean-reversion model)."""
        next_state = copy.deepcopy(state)
        for entity_metrics in next_state.values():
            if not isinstance(entity_metrics, dict):
                continue
            # Utilization decays slightly toward 50% each step (mean reversion)
            util = float(entity_metrics.get("utilization_pct", 50.0))
            entity_metrics["utilization_pct"] = util + 0.05 * (50.0 - util)
            # Latency also reverts toward baseline
            latency = float(entity_metrics.get("latency_ms", 5.0))
            entity_metrics["latency_ms"] = max(0.5, latency + 0.03 * (5.0 - latency))
        return next_state

    def _compute_summary(
        self,
        action: ProposedAction,
        predicted_states: list[dict[str, dict[str, float]]],
    ) -> dict[str, Any]:
        """Aggregate predicted states into a risk summary."""
        if not predicted_states:
            return {
                "max_utilization": 0.0,
                "risk_score": 0.0,
                "safe": True,
                "rationale": "No simulation steps produced.",
            }

        max_util = 0.0
        max_latency = 0.0
        max_pktloss = 0.0
        sla_violations = 0

        for step_state in predicted_states:
            for entity_metrics in step_state.values():
                if not isinstance(entity_metrics, dict):
                    continue
                util = float(entity_metrics.get("utilization_pct", 0.0))
                latency = float(entity_metrics.get("latency_ms", 0.0))
                pktloss = float(entity_metrics.get("packet_loss_pct", 0.0))

                max_util = max(max_util, util)
                max_latency = max(max_latency, latency)
                max_pktloss = max(max_pktloss, pktloss)

                if pktloss > 1.0 or latency > 50.0:
                    sla_violations += 1

        util_risk = min(1.0, max(0.0, (max_util - 50.0) / 50.0))
        latency_risk = min(1.0, max(0.0, (max_latency - 10.0) / 90.0))
        pktloss_risk = min(1.0, max_pktloss / 5.0)
        sla_risk = min(1.0, sla_violations / max(len(predicted_states) * 2, 1))

        risk_score = (
            _RISK_WEIGHTS["utilization_pct"] * util_risk
            + _RISK_WEIGHTS["latency_ms"] * latency_risk
            + _RISK_WEIGHTS["packet_loss_pct"] * pktloss_risk
            + _RISK_WEIGHTS["sla"] * sla_risk
        )
        safe = max_util < _SAFE_UTIL_LIMIT

        rationale = (
            f"Digital twin: action={action.action_type}, "
            f"max_util={max_util:.1f}%, risk_score={risk_score:.3f}, safe={safe}."
        )

        return {
            "max_utilization": round(max_util, 4),
            "risk_score": round(risk_score, 6),
            "safe": safe,
            "rationale": rationale,
        }
