from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from src.models.schemas import ActionResult, ProposedAction
from src.simulator.engine import SimulationEngine
from src.utils.logger import logger


class ActorAgent:
    """Executes approved actions on simulation engine with rollback protection."""

    def __init__(self, simulation_engine: SimulationEngine) -> None:
        """Initialize action executor with runtime safety guards."""
        self.simulation_engine = simulation_engine
        self.action_history: list[dict[str, Any]] = []
        self.rollback_store: dict[str, dict[str, Any]] = {}
        self.active_monitors: dict[str, asyncio.Task[None]] = {}
        self.kill_switch_active: bool = False
        self.actions_in_window: list[tuple[datetime, str]] = []
        self._blocked_regions: set[str] = set()
        self._last_rollback_token: str | None = None

    def execute(self, action: ProposedAction) -> ActionResult:
        """Execute an action, attach rollback/monitoring, and return immediate result."""
        if self._is_kill_switch_blocking(action):
            return self._failed_result(action.id, "ineffective", "Kill switch is active for this target.")

        if self._is_rate_limited():
            return self._failed_result(action.id, "ineffective", "Rate limit exceeded: >3 actions in 10 minutes.")

        pre_snapshot = self.simulation_engine.get_current_state_dict()
        affected_entities = self._affected_entities(action)
        pre_metrics = self._capture_entity_metrics(pre_snapshot, affected_entities)

        rollback_token = str(uuid.uuid4())
        self.rollback_store[rollback_token] = {
            "action_id": action.id,
            "action_type": action.action_type,
            "pre_state": pre_snapshot,
            "affected_entities": list(affected_entities),
            "engine_rollback_token": None,
        }

        success, engine_rollback_token = self._apply_to_engine(action)
        post_snapshot = self.simulation_engine.get_current_state_dict()
        post_metrics = self._capture_entity_metrics(post_snapshot, affected_entities)

        actual_rollback_token = engine_rollback_token or rollback_token
        rollback_available = success and action.action_type not in {"create_ticket", "escalate"}
        if rollback_available:
            self.rollback_store[actual_rollback_token] = self.rollback_store.pop(rollback_token)
            self.rollback_store[actual_rollback_token]["engine_rollback_token"] = engine_rollback_token
            self._last_rollback_token = actual_rollback_token
        else:
            self.rollback_store.pop(rollback_token, None)
            actual_rollback_token = None

        if success:
            self.actions_in_window.append((datetime.now(timezone.utc), action.id))
            self._start_monitoring(
                action=action,
                rollback_token=actual_rollback_token,
                pre_metrics=pre_metrics,
            )

        result = ActionResult(
            action_id=action.id,
            success=success,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            rollback_available=rollback_available,
            rollback_token=actual_rollback_token,
            outcome="partially_effective" if success else "ineffective",
        )

        self.action_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action.model_dump(),
                "result": result.model_dump(),
                "rollback_token": actual_rollback_token,
            }
        )
        return result

    def rollback(self, rollback_token: str) -> bool:
        """Rollback a previously executed action."""
        if rollback_token not in self.rollback_store:
            return False

        state = self.rollback_store[rollback_token]
        engine_token = state.get("engine_rollback_token") or rollback_token
        success = bool(self.simulation_engine.rollback_action(engine_token))
        if not success:
            return False

        action_id = str(state.get("action_id", ""))
        task = self.active_monitors.pop(action_id, None)
        if task and not task.done():
            task.cancel()

        del self.rollback_store[rollback_token]
        return True

    def kill_switch(self, region: str | None = None) -> None:
        """Activate kill switch globally or for a specific region and rollback recent action."""
        if region:
            self._blocked_regions.add(region)
            logger.warning("Kill switch activated for region={}", region)
        else:
            self.kill_switch_active = True
            logger.warning("Global kill switch activated.")

        for entry in reversed(self.action_history):
            result = entry.get("result", {})
            action = entry.get("action", {})
            token = entry.get("rollback_token")
            if not token:
                continue
            if action.get("action_type") in {"create_ticket", "escalate"}:
                continue
            if not result.get("success", False):
                continue
            if self.rollback(str(token)):
                logger.warning("Kill switch rollback succeeded for action_id={}", action.get("id"))
            break

    def resume_operations(self, region: str | None = None) -> None:
        """Resume operations globally or for a blocked region."""
        if region:
            self._blocked_regions.discard(region)
            logger.info("Operations resumed for region={}", region)
            return
        self.kill_switch_active = False
        self._blocked_regions.clear()
        logger.info("Global operations resumed.")

    def get_action_history(self) -> list[ActionResult]:
        """Return action result history."""
        results: list[ActionResult] = []
        for entry in self.action_history:
            payload = entry.get("result")
            if isinstance(payload, dict):
                try:
                    results.append(ActionResult(**payload))
                except Exception:
                    continue
        return results

    def _apply_to_engine(self, action: ProposedAction) -> tuple[bool, str | None]:
        if action.action_type == "create_ticket":
            logger.info("Ticket created for action_id={} target={}", action.id, action.target_node or action.target_link)
            return True, None

        if action.action_type == "escalate":
            logger.warning("Escalation created for action_id={} target={}", action.id, action.target_node or action.target_link)
            return True, None

        if action.action_type == "config_rollback":
            token = str(action.parameters.get("rollback_token") or self._last_rollback_token or "")
            if not token:
                return False, None
            return bool(self.simulation_engine.rollback_action(token)), token

        payload = self._map_action_payload(action)
        response = self.simulation_engine.apply_action(payload)
        success = bool(response.get("success", False))
        token = response.get("rollback_token")
        if isinstance(token, str) and token:
            return success, token
        return success, None

    def _map_action_payload(self, action: ProposedAction) -> dict[str, Any]:
        params = dict(action.parameters)
        if action.action_type == "reroute":
            source_node = params.get("source_node")
            target_node = params.get("target_node")
            if (not source_node or not target_node) and action.target_link and "-" in action.target_link:
                parts = action.target_link.split("-")
                if len(parts) >= 2:
                    source_node = source_node or parts[0]
                    target_node = target_node or parts[1]
            return {
                "action_type": "reroute",
                "source_node": source_node,
                "target_node": target_node,
                "new_weight": float(params.get("new_weight", 8.0)),
            }

        if action.action_type == "rate_limit":
            return {
                "action_type": "rate_limit",
                "target_link": params.get("target_link") or action.target_link,
                "limit_pct": float(params.get("limit_pct", 70.0)),
            }

        if action.action_type == "scale_capacity":
            return {
                "action_type": "scale_capacity",
                "target_link": params.get("target_link") or action.target_link,
                "new_capacity_gbps": float(params.get("new_capacity_gbps", 100.0)),
            }

        return {"action_type": action.action_type, **params}

    def _start_monitoring(
        self, action: ProposedAction, rollback_token: str | None, pre_metrics: dict[str, float]
    ) -> None:
        if not rollback_token:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        task = loop.create_task(
            self._monitor_action(action_id=action.id, rollback_token=rollback_token, pre_metrics=pre_metrics)
        )
        self.active_monitors[action.id] = task

    async def _monitor_action(
        self, action_id: str, rollback_token: str, pre_metrics: dict[str, float]
    ) -> None:
        for _ in range(10):
            await asyncio.sleep(60)
            snapshot = self.simulation_engine.get_current_state_dict()
            current_metrics = self._capture_metrics_by_keys(snapshot, list(pre_metrics.keys()))
            if self._has_worsened_by_20pct(pre_metrics, current_metrics):
                logger.warning(
                    "Auto-rollback triggered for action_id={} rollback_token={}", action_id, rollback_token
                )
                self.rollback(rollback_token)
                break

        self.active_monitors.pop(action_id, None)

    def _has_worsened_by_20pct(
        self, pre_metrics: dict[str, float], current_metrics: dict[str, float]
    ) -> bool:
        for key, pre in pre_metrics.items():
            cur = current_metrics.get(key)
            if cur is None:
                continue

            metric_name = key.split(".", 1)[1] if "." in key else key
            if metric_name in {"latency_ms", "utilization_pct", "packet_loss_pct", "cpu_pct", "memory_pct", "temperature_c", "buffer_drops"}:
                if pre > 0 and cur > pre * 1.2:
                    return True
            elif metric_name in {"throughput_gbps"}:
                if pre > 0 and cur < pre * 0.8:
                    return True
        return False

    def _capture_entity_metrics(
        self, snapshot: dict[str, Any], entities: set[str]
    ) -> dict[str, float]:
        output: dict[str, float] = {}
        nodes = snapshot.get("nodes", {})
        links = snapshot.get("links", {})

        for entity in entities:
            metrics = nodes.get(entity) if entity in nodes else links.get(entity)
            if not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    output[f"{entity}.{metric}"] = float(value)
        return output

    def _capture_metrics_by_keys(
        self, snapshot: dict[str, Any], keys: list[str]
    ) -> dict[str, float]:
        nodes = snapshot.get("nodes", {})
        links = snapshot.get("links", {})
        output: dict[str, float] = {}
        for key in keys:
            if "." not in key:
                continue
            entity, metric = key.split(".", 1)
            pool = nodes if entity in nodes else links
            metrics = pool.get(entity, {})
            value = metrics.get(metric) if isinstance(metrics, dict) else None
            if isinstance(value, (int, float)):
                output[key] = float(value)
        return output

    def _affected_entities(self, action: ProposedAction) -> set[str]:
        entities: set[str] = set()
        if action.target_node:
            entities.add(action.target_node)
        if action.target_link:
            entities.add(action.target_link)
            if "-" in action.target_link:
                left, right = action.target_link.split("-")[:2]
                entities.add(left)
                entities.add(right)

        for key in ("from_link", "to_link", "target_link"):
            value = action.parameters.get(key)
            if isinstance(value, str) and value:
                entities.add(value)
                if "-" in value:
                    left, right = value.split("-")[:2]
                    entities.add(left)
                    entities.add(right)

        for key in ("source_node", "target_node"):
            value = action.parameters.get(key)
            if isinstance(value, str) and value:
                entities.add(value)

        return entities

    def _is_kill_switch_blocking(self, action: ProposedAction) -> bool:
        if self.kill_switch_active:
            return True
        if not self._blocked_regions:
            return False

        regions = self._target_regions(action)
        return any(region in self._blocked_regions for region in regions)

    def _target_regions(self, action: ProposedAction) -> set[str]:
        topology = getattr(self.simulation_engine, "_topology", None)
        if topology is None:
            return set()

        node_regions: set[str] = set()
        nodes = topology.get_all_nodes()
        node_index = {node["node_id"]: node for node in nodes}
        entities = self._affected_entities(action)
        for entity in entities:
            if entity in node_index:
                node_regions.add(str(node_index[entity].get("region", "")))
            elif "-" in entity:
                left, right = entity.split("-")[:2]
                if left in node_index:
                    node_regions.add(str(node_index[left].get("region", "")))
                if right in node_index:
                    node_regions.add(str(node_index[right].get("region", "")))
        return {region for region in node_regions if region}

    def _is_rate_limited(self) -> bool:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=10)
        self.actions_in_window = [(ts, action_id) for ts, action_id in self.actions_in_window if ts >= cutoff]
        return len(self.actions_in_window) >= 3

    def _failed_result(self, action_id: str, outcome: str, reason: str) -> ActionResult:
        logger.warning("Action refused action_id={} reason={}", action_id, reason)
        return ActionResult(
            action_id=action_id,
            success=False,
            pre_metrics={},
            post_metrics={},
            rollback_available=False,
            rollback_token=None,
            outcome=outcome,  # type: ignore[arg-type]
        )
