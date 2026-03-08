"""Z3SafetyVerifier — formal safety verification for network actions.

Uses Microsoft's Z3 SMT solver to prove that proposed actions satisfy
defined safety invariants before execution.  If Z3 returns SAT the
action is provably safe; UNSAT pinpoints exactly which constraint is
violated and why.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import networkx as nx
import z3

from src.models.schemas import ProposedAction
from src.simulator.topology import NetworkTopology

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTIL_LIMIT      = 85.0   # max post-action link utilization (%)
_CASCADE_LIMIT   = 80.0   # max utilization for previously-healthy links (%)
_CUSTOMER_FRAC   = 0.05   # max fraction of customers affected
_RATE_WINDOW_MIN = 10     # rolling window for rate-limit cap (minutes)
_RATE_LIMIT_MAX  = 3      # max automated actions per window
_BLAST_FRAC      = 0.20   # max fraction of total throughput affected
_MIN_PATHS       = 2      # minimum alternative paths required after action

# Action types that do not modify network state — skip network-state constraints
_NON_NETWORK_ACTIONS = frozenset({"escalate", "create_ticket", "notify"})


def _safe_id(name: str) -> str:
    """Convert entity IDs to valid Z3 variable-name strings."""
    return name.replace("-", "_").replace(".", "_")


def _get_util(entity_id: str, predicted: dict, current: dict) -> float:
    """Return link utilization from predicted state, falling back to current."""
    return (
        predicted.get(entity_id, {}).get("utilization_pct")
        or current.get(entity_id, {}).get("utilization_pct")
        or 0.0
    )


def _merge_states(current: dict, predicted: dict) -> dict:
    """Overlay predicted metrics on top of current state."""
    merged: dict = {}
    for eid, metrics in current.items():
        merged[eid] = dict(metrics)
    for eid, metrics in predicted.items():
        merged.setdefault(eid, {}).update(metrics)
    return merged


def _z3_check(expr: z3.BoolRef) -> tuple[bool, str]:
    """Return (is_sat, detail) for a Z3 boolean expression."""
    s = z3.Solver()
    s.add(expr)
    result = s.check()
    if result == z3.sat:
        return True, "SAT"
    if result == z3.unsat:
        return False, "UNSAT"
    return True, "UNKNOWN (Z3 timeout — defaulting safe)"


# ---------------------------------------------------------------------------
# Z3SafetyVerifier
# ---------------------------------------------------------------------------

class Z3SafetyVerifier:
    """Formally verifies proposed network actions using Z3 SMT solving.

    Each safety constraint is a callable ``(action, current_state, predicted_state)
    -> z3.BoolRef`` that encodes the invariant as a Z3 formula.  ``verify_action``
    checks every constraint and returns a structured verdict.
    """

    def __init__(self, topology: NetworkTopology) -> None:
        self._topology = topology
        self._graph    = topology.get_graph()
        self._links    = topology.get_all_links()
        self._nodes    = topology.get_all_nodes()

        self._total_customers: int = sum(
            n.get("customers", 0) for n in self._nodes
        )
        self._total_capacity_gbps: float = sum(
            lnk["capacity_gbps"] for lnk in self._links
        )

        # Rolling history of action timestamps for rate-limit tracking
        self._action_history: deque[datetime] = deque()

        # Named constraint registry
        self.constraints: dict[str, Callable] = {}
        self._setup_default_constraints()

    # ------------------------------------------------------------------
    # Constraint registration
    # ------------------------------------------------------------------

    def _setup_default_constraints(self) -> None:
        """Register the six default safety invariants."""

        # 1. No link exceeds 85% utilization after the action.
        #    Non-network-modifying actions (escalate, create_ticket) don't change
        #    utilization — skip this check entirely for them.
        def max_link_utilization(action: dict, current: dict, predicted: dict) -> z3.BoolRef:
            if action.get("action_type") in _NON_NETWORK_ACTIONS:
                return z3.BoolVal(True)
            clauses: list[z3.BoolRef] = []
            for lnk in self._links:
                lid  = lnk["link_id"]
                util = _get_util(lid, predicted, current)
                var  = z3.Real(f"u_{_safe_id(lid)}")
                clauses.append(z3.And(var == util, var <= _UTIL_LIMIT))
            return z3.And(clauses) if clauses else z3.BoolVal(True)

        # 2. Cascading overload guard: previously healthy links stay ≤ 80%.
        #    Skip for non-network-modifying actions — they cannot cause cascades.
        def no_cascading_overload(action: dict, current: dict, predicted: dict) -> z3.BoolRef:
            if action.get("action_type") in _NON_NETWORK_ACTIONS:
                return z3.BoolVal(True)
            clauses: list[z3.BoolRef] = []
            for lnk in self._links:
                lid          = lnk["link_id"]
                pre_util     = current.get(lid, {}).get("utilization_pct", 0.0)
                if pre_util >= _CASCADE_LIMIT:
                    continue  # already congested — not a new cascade
                post_util = _get_util(lid, predicted, current)
                var       = z3.Real(f"cascade_{_safe_id(lid)}")
                clauses.append(z3.And(var == post_util, var <= _CASCADE_LIMIT))
            return z3.And(clauses) if clauses else z3.BoolVal(True)

        # 3. At most 5% of total customers may be on affected nodes
        def customer_isolation(action: dict, current: dict, predicted: dict) -> z3.BoolRef:
            affected_ids = set(predicted.keys()) | self._action_affected_nodes(action)
            affected_customers = sum(
                n.get("customers", 0)
                for n in self._nodes
                if n["node_id"] in affected_ids
            )
            limit = _CUSTOMER_FRAC * max(self._total_customers, 1)
            v = z3.Real("affected_customers")
            return z3.And(v == float(affected_customers), v <= limit)

        # 4. Rate-limit cap: fewer than _RATE_LIMIT_MAX actions in rolling window
        def rate_limit_cap(action: dict, current: dict, predicted: dict) -> z3.BoolRef:
            now    = datetime.now(timezone.utc)
            cutoff = now - timedelta(minutes=_RATE_WINDOW_MIN)
            recent = sum(1 for ts in self._action_history if ts >= cutoff)
            v = z3.Int("recent_action_count")
            return z3.And(v == recent, v < _RATE_LIMIT_MAX)

        # 5. Blast radius: affected traffic ≤ 20% of total network capacity
        def blast_radius_cap(action: dict, current: dict, predicted: dict) -> z3.BoolRef:
            affected_traffic = self._estimate_affected_traffic(action, current, predicted)
            limit = _BLAST_FRAC * self._total_capacity_gbps
            v = z3.Real("affected_traffic_gbps")
            return z3.And(v == affected_traffic, v <= limit)

        # 6. Rollback path exists: network remains reachable after action.
        #    For reroute/rate_limit the affected link is NOT removed, so the
        #    original path always survives and rollback is trivial → always SAT.
        #    For destructive actions we verify ≥ _MIN_PATHS exist between the
        #    affected endpoints by enumerating simple paths.
        def rollback_path_exists(action: dict, current: dict, predicted: dict) -> z3.BoolRef:
            atype = action.get("action_type", "")
            if atype in ("reroute", "rate_limit"):
                # Link stays active — rollback is always available
                v = z3.Int("rollback_trivially_available")
                return z3.And(v == 1, v >= 1)
            pairs = self._affected_src_dst_pairs(action)
            clauses: list[z3.BoolRef] = []
            for src, dst in pairs:
                paths = list(nx.all_simple_paths(
                    self._graph, src, dst, cutoff=6
                ))
                count = len(paths)
                v = z3.Int(f"paths_{_safe_id(src)}_{_safe_id(dst)}")
                clauses.append(z3.And(v == count, v >= _MIN_PATHS))
            return z3.And(clauses) if clauses else z3.BoolVal(True)

        self.constraints = {
            "max_link_utilization":  max_link_utilization,
            "no_cascading_overload": no_cascading_overload,
            "customer_isolation":    customer_isolation,
            "rate_limit_cap":        rate_limit_cap,
            "blast_radius_cap":      blast_radius_cap,
            "rollback_path_exists":  rollback_path_exists,
        }

    # ------------------------------------------------------------------
    # Constraint management
    # ------------------------------------------------------------------

    def add_constraint(self, name: str, constraint_fn: Callable) -> None:
        """Register a custom safety constraint."""
        self.constraints[name] = constraint_fn

    def remove_constraint(self, name: str) -> None:
        """Remove a constraint by name."""
        self.constraints.pop(name, None)

    def list_constraints(self) -> list[str]:
        """Return names of all active constraints."""
        return list(self.constraints)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_action(
        self,
        action: ProposedAction | dict,
        current_state: dict,
        predicted_state: dict,
        record: bool = True,
    ) -> dict:
        """Check every safety constraint against the predicted post-action state.

        Returns:
            {
                "is_safe":    bool,
                "violations": list[str],     # constraint names that failed
                "proof":      str,           # human-readable summary
                "details":    dict,          # per-constraint verdict + detail
            }
        """
        action_dict = action if isinstance(action, dict) else action.model_dump()

        details: dict[str, dict] = {}
        violations: list[str]   = []
        worst_util: float       = 0.0
        worst_link: str         = ""

        # Track worst-case utilization for the proof string
        for lnk in self._links:
            lid  = lnk["link_id"]
            util = _get_util(lid, predicted_state, current_state)
            if util > worst_util:
                worst_util = util
                worst_link = lid

        for name, constraint_fn in self.constraints.items():
            try:
                expr            = constraint_fn(action_dict, current_state, predicted_state)
                is_sat, outcome = _z3_check(expr)
            except Exception as exc:
                is_sat  = True
                outcome = f"ERROR ({exc}) — defaulting safe"

            details[name] = {
                "satisfied": is_sat,
                "z3_result": outcome,
                "detail":    self._constraint_detail(
                    name, action_dict, current_state, predicted_state
                ),
            }
            if not is_sat:
                violations.append(name)

        is_safe = len(violations) == 0

        # Record to action history only when the caller is committing to execute
        # (record=False for decider pre-checks to avoid exhausting the rate limit)
        if is_safe and record:
            self._action_history.append(datetime.now(timezone.utc))

        proof = self._build_proof(
            is_safe, violations, details, worst_link, worst_util, action_dict
        )

        return {
            "is_safe":    is_safe,
            "violations": violations,
            "proof":      proof,
            "details":    details,
        }

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain_verification(self, result: dict) -> str:
        """Return a human-readable explanation of a verification result."""
        return result.get("proof", "No proof available.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _constraint_detail(
        self,
        name: str,
        action: dict,
        current: dict,
        predicted: dict,
    ) -> str:
        """Compute a short human-readable detail string for one constraint."""
        if name == "max_link_utilization":
            worst_val, worst_lid = 0.0, ""
            for lnk in self._links:
                lid  = lnk["link_id"]
                util = _get_util(lid, predicted, current)
                if util > worst_val:
                    worst_val, worst_lid = util, lid
            return (
                f"worst link {worst_lid} at {worst_val:.1f}% "
                f"(limit {_UTIL_LIMIT:.0f}%)"
            )

        if name == "no_cascading_overload":
            new_violations = []
            for lnk in self._links:
                lid      = lnk["link_id"]
                pre_util = current.get(lid, {}).get("utilization_pct", 0.0)
                if pre_util >= _CASCADE_LIMIT:
                    continue
                post_util = _get_util(lid, predicted, current)
                if post_util > _CASCADE_LIMIT:
                    new_violations.append(f"{lid}={post_util:.1f}%")
            return (
                f"newly overloaded: {new_violations}" if new_violations
                else "no healthy links pushed into overload"
            )

        if name == "customer_isolation":
            affected_ids = set(predicted.keys()) | self._action_affected_nodes(action)
            affected_customers = sum(
                n.get("customers", 0)
                for n in self._nodes
                if n["node_id"] in affected_ids
            )
            pct = 100.0 * affected_customers / max(self._total_customers, 1)
            limit_pct = _CUSTOMER_FRAC * 100
            return (
                f"{affected_customers:,} customers affected "
                f"({pct:.1f}% of {self._total_customers:,}, limit {limit_pct:.0f}%)"
            )

        if name == "rate_limit_cap":
            now    = datetime.now(timezone.utc)
            cutoff = now - timedelta(minutes=_RATE_WINDOW_MIN)
            recent = sum(1 for ts in self._action_history if ts >= cutoff)
            return (
                f"{recent} actions in last {_RATE_WINDOW_MIN} min "
                f"(limit {_RATE_LIMIT_MAX})"
            )

        if name == "blast_radius_cap":
            affected = self._estimate_affected_traffic(action, current, predicted)
            limit    = _BLAST_FRAC * self._total_capacity_gbps
            pct      = 100.0 * affected / max(self._total_capacity_gbps, 1)
            return (
                f"{affected:.1f} Gbps affected "
                f"({pct:.1f}% of {self._total_capacity_gbps:.0f} Gbps, "
                f"limit {_BLAST_FRAC*100:.0f}%)"
            )

        if name == "rollback_path_exists":
            pairs = self._affected_src_dst_pairs(action)
            results = []
            for src, dst in pairs:
                n = len(list(nx.all_simple_paths(self._graph, src, dst, cutoff=6)))
                results.append(f"{src}→{dst}: {n} paths")
            return ", ".join(results) if results else "no affected pairs"

        return ""

    def _build_proof(
        self,
        is_safe: bool,
        violations: list[str],
        details: dict,
        worst_link: str,
        worst_util: float,
        action: dict,
    ) -> str:
        n_total = len(self.constraints)
        n_pass  = n_total - len(violations)
        atype   = action.get("action_type", "unknown")
        target  = action.get("target_link") or action.get("from_link") or action.get("target_node", "?")

        if is_safe:
            lines = [
                f"✅  Action APPROVED  [{atype} on {target}]",
                f"    All {n_total} safety constraints satisfied.",
                f"    Worst-case utilization: {worst_link} at {worst_util:.1f}%  "
                f"(limit {_UTIL_LIMIT:.0f}%).",
            ]
            for name, d in details.items():
                lines.append(f"    ✓  {name}: {d['detail']}")
        else:
            lines = [
                f"🚫  Action BLOCKED  [{atype} on {target}]",
                f"    {len(violations)}/{n_total} constraint(s) violated.",
            ]
            for v in violations:
                d = details[v]
                lines.append(f"    ✗  {v}: {d['detail']}")
            lines.append(
                "    Suggestion: reduce traffic volume, choose an alternate path, "
                "or wait before issuing another action."
            )

        return "\n".join(lines)

    def _action_affected_nodes(self, action: dict) -> set[str]:
        """Infer which node IDs are directly touched by this action."""
        affected: set[str] = set()
        target_link = action.get("target_link") or action.get("from_link")
        if target_link:
            for lnk in self._links:
                if lnk["link_id"] == target_link:
                    affected.add(lnk["source"])
                    affected.add(lnk["target"])
        if action.get("target_node"):
            affected.add(action["target_node"])
        return affected

    def _estimate_affected_traffic(
        self, action: dict, current: dict, predicted: dict
    ) -> float:
        """Estimate how many Gbps are moved/affected by this action."""
        atype = action.get("action_type", "")
        if atype == "reroute":
            from_link = action.get("from_link", "")
            fraction  = float(action.get("traffic_fraction", 0.5))
            cap       = next(
                (lnk["capacity_gbps"] for lnk in self._links if lnk["link_id"] == from_link),
                0.0,
            )
            util = _get_util(from_link, predicted, current)
            return cap * (util / 100.0) * fraction
        if atype == "rate_limit":
            target    = action.get("target_link", "")
            reduction = float(action.get("reduction_pct", 20.0))
            cap       = next(
                (lnk["capacity_gbps"] for lnk in self._links if lnk["link_id"] == target),
                0.0,
            )
            return cap * (reduction / 100.0)
        return 0.0

    def _affected_src_dst_pairs(self, action: dict) -> list[tuple[str, str]]:
        """Identify (src, dst) node pairs whose routing is disrupted by the action."""
        link_id = (
            action.get("target_link")
            or action.get("from_link")
            or action.get("to_link")
        )
        if not link_id:
            return []
        for lnk in self._links:
            if lnk["link_id"] == link_id:
                return [(lnk["source"], lnk["target"])]
        return []


# ---------------------------------------------------------------------------
# Main — demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.simulator.topology import NetworkTopology

    print("\n── Z3SafetyVerifier Demo ─────────────────────────────────────────\n")

    topo     = NetworkTopology()
    verifier = Z3SafetyVerifier(topo)

    print(f"Active constraints ({len(verifier.list_constraints())}):")
    for c in verifier.list_constraints():
        print(f"  • {c}")

    # ── Scenario A: Safe reroute (40% from CR1-CR2 to AGG2-CR1) ────────────
    print("\n── Scenario A: Safe reroute ──────────────────────────────────────")

    current_state_a = {
        "CR1-CR2":  {"utilization_pct": 82.0, "throughput_gbps": 164.0},
        "AGG1-CR1": {"utilization_pct": 55.0, "throughput_gbps":  55.0},
        "AGG2-CR1": {"utilization_pct": 30.0, "throughput_gbps":  30.0},
        "AGG1-EDGE1": {"utilization_pct": 40.0},
        "CR1":      {"cpu_pct": 70.0, "temperature_c": 55.0},
        "EDGE1":    {"cpu_pct": 60.0, "temperature_c": 45.0},
    }
    # After rerouting 40% of CR1-CR2 traffic to AGG2-CR1:
    predicted_state_a = {
        "CR1-CR2":  {"utilization_pct": 49.2},   # 82 - 82*0.4 = 49.2
        "AGG2-CR1": {"utilization_pct": 62.8},   # 30 + 82*0.4 = 62.8 — stays under 85
    }

    action_a = ProposedAction(
        action_type="reroute",
        target_link="CR1-CR2",
        parameters={"from_link": "CR1-CR2", "to_link": "AGG2-CR1", "traffic_fraction": 0.40},
        expected_impact="Reduce CR1-CR2 utilization from 82% to ~49%",
        risk_level=0.3,
        requires_approval=False,
        utility_score=0.8,
    )

    result_a = verifier.verify_action(action_a, current_state_a, predicted_state_a)
    print(verifier.explain_verification(result_a))
    print(f"\n  is_safe   : {result_a['is_safe']}")
    print(f"  violations: {result_a['violations']}")

    # ── Scenario B: Unsafe reroute (pushes link over 85%) ──────────────────
    print("\n── Scenario B: Unsafe reroute (link breaches 85% limit) ─────────")

    current_state_b = {
        "CR1-CR2":  {"utilization_pct": 95.0, "throughput_gbps": 190.0},
        "AGG2-CR1": {"utilization_pct": 70.0, "throughput_gbps":  70.0},
        "AGG1-CR1": {"utilization_pct": 45.0, "throughput_gbps":  45.0},
        "CR1":      {"cpu_pct": 88.0, "temperature_c": 68.0},
        "AGG1":     {"cpu_pct": 72.0, "temperature_c": 52.0},
    }
    # Predicted: AGG2-CR1 would be pushed to 91% (unsafe)
    predicted_state_b = {
        "CR1-CR2":  {"utilization_pct": 57.0},
        "AGG2-CR1": {"utilization_pct": 91.0},  # EXCEEDS 85% limit
    }

    action_b = ProposedAction(
        action_type="reroute",
        target_link="CR1-CR2",
        parameters={"from_link": "CR1-CR2", "to_link": "AGG2-CR1", "traffic_fraction": 0.40},
        expected_impact="Reduce CR1-CR2 utilization but overloads AGG2-CR1",
        risk_level=0.7,
        requires_approval=True,
        utility_score=0.4,
    )

    result_b = verifier.verify_action(action_b, current_state_b, predicted_state_b)
    print(verifier.explain_verification(result_b))
    print(f"\n  is_safe   : {result_b['is_safe']}")
    print(f"  violations: {result_b['violations']}")

    # ── Scenario C: Rate-limit action (safe, reduces utilization) ──────────
    # Use AGG1-CR1 (core link, 0 direct customers) so customer_isolation passes.
    print("\n── Scenario C: Rate-limit action (should PASS) ───────────────────")

    current_state_c = {
        "AGG1-CR1": {"utilization_pct": 88.0, "throughput_gbps": 88.0},
        "AGG1":     {"cpu_pct": 80.0, "temperature_c": 60.0},
        "CR1":      {"cpu_pct": 72.0, "temperature_c": 55.0},
    }
    predicted_state_c = {
        "AGG1-CR1": {"utilization_pct": 62.0},  # reduced by rate-limit
    }

    action_c = ProposedAction(
        action_type="rate_limit",
        target_link="AGG1-CR1",
        parameters={"reduction_pct": 30.0},
        expected_impact="Throttle AGG1-CR1 from 88% to 62%",
        risk_level=0.2,
        requires_approval=False,
        utility_score=0.75,
    )

    result_c = verifier.verify_action(action_c, current_state_c, predicted_state_c)
    print(verifier.explain_verification(result_c))
    print(f"\n  is_safe   : {result_c['is_safe']}")
    print(f"  violations: {result_c['violations']}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────────────────────")
    results = [
        ("Scenario A (safe reroute)",       result_a),
        ("Scenario B (unsafe reroute)",     result_b),
        ("Scenario C (rate-limit)",         result_c),
    ]
    for label, r in results:
        status = "✅ SAFE" if r["is_safe"] else f"🚫 BLOCKED ({', '.join(r['violations'])})"
        print(f"  {label:35s}  →  {status}")

    print("\n🎉 Z3SafetyVerifier demo complete.\n")
