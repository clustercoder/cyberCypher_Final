"""CausalEngine — Digital Twin causal reasoning for the CyberCypher 5.0 agent.

Combines structural knowledge (topology) with data-driven causal discovery to
support root-cause analysis and counterfactual ("what if") reasoning.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src.models.schemas import Anomaly, Hypothesis
from src.simulator.topology import NetworkTopology

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps raw metric column names → short causal-variable suffixes
_METRIC_SHORT: dict[str, str] = {
    "utilization_pct": "utilization",
    "packet_loss_pct": "packet_loss",
    "latency_ms": "latency",
    "cpu_pct": "cpu",
    "temperature_c": "temperature",
    "buffer_drops": "buffer_drops",
    "memory_pct": "memory",
    "throughput_gbps": "throughput",
}

# Node type → tier (lower = closer to the internet/upstream)
_TIER: dict[str, int] = {
    "peering": 0,
    "core": 1,
    "aggregation": 2,
    "edge": 3,
}

# Node-type colors for visualization
_NODE_VIZ_COLORS = {
    "anomalous": "#e74c3c",   # red
    "root_cause": "#f39c12",  # orange
    "normal": "#2ecc71",      # green
}

# Minimum lagged-correlation threshold to add a learned edge
_CORRELATION_THRESHOLD = 0.6
_MAX_LAG_STEPS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short(metric_name: str) -> str:
    """Normalize a raw metric name to its short causal-variable suffix."""
    return _METRIC_SHORT.get(metric_name, metric_name)


def _causal_var(entity_id: str, metric_name: str) -> str:
    """Return the causal graph node name for an entity + metric pair."""
    return f"{entity_id}_{_short(metric_name)}"


def _anomaly_to_causal_var(anomaly: Anomaly) -> Optional[str]:
    """Convert an Anomaly to its causal graph variable name."""
    if anomaly.link_id:
        return _causal_var(anomaly.link_id, anomaly.metric_name)
    if anomaly.node_id:
        return _causal_var(anomaly.node_id, anomaly.metric_name)
    return None


# ---------------------------------------------------------------------------
# CausalEngine
# ---------------------------------------------------------------------------

class CausalEngine:
    """Digital Twin causal reasoning engine.

    Combines structural causal knowledge derived from the network topology
    with data-driven edges discovered from telemetry.  Supports root-cause
    analysis (walk backwards through the causal graph) and counterfactual
    simulation (propagate actions forward).
    """

    def __init__(self, topology: NetworkTopology) -> None:
        self._topology = topology
        self._graph_topo: nx.DiGraph = topology.get_graph()
        self._node_meta: dict[str, dict] = {
            n: dict(self._graph_topo.nodes[n]) for n in self._graph_topo.nodes
        }

        self.structural_graph: nx.DiGraph = nx.DiGraph()
        self.learned_graph: nx.DiGraph = nx.DiGraph()
        self.combined_graph: nx.DiGraph = nx.DiGraph()

        self.build_structural_graph()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _node_tier(self, node_id: str) -> int:
        ntype = self._node_meta.get(node_id, {}).get("node_type", "edge")
        return _TIER.get(ntype, 3)

    def _link_direction(self, link: dict) -> tuple[str, str]:
        """Return (upstream_node, downstream_node) based on network tier."""
        src, tgt = link["source"], link["target"]
        if self._node_tier(src) <= self._node_tier(tgt):
            return src, tgt
        return tgt, src

    def _add_edge(
        self,
        graph: nx.DiGraph,
        src_var: str,
        tgt_var: str,
        strength: float,
        edge_type: str = "structural",
    ) -> None:
        """Add a causal edge, keeping max strength if edge already exists."""
        if graph.has_edge(src_var, tgt_var):
            if strength <= graph.edges[src_var, tgt_var].get("causal_strength", 0.0):
                return
        graph.add_node(src_var)
        graph.add_node(tgt_var)
        graph.add_edge(src_var, tgt_var, causal_strength=strength, edge_type=edge_type)

    def _merge_graphs(self) -> None:
        """Build combined_graph = structural_graph ∪ learned_graph (max strength)."""
        combined = nx.DiGraph()
        for src, tgt, data in self.structural_graph.edges(data=True):
            combined.add_edge(src, tgt, **data)
        for src, tgt, data in self.learned_graph.edges(data=True):
            if combined.has_edge(src, tgt):
                existing = combined.edges[src, tgt].get("causal_strength", 0.0)
                new_str = data.get("causal_strength", 0.0)
                if new_str > existing:
                    combined.edges[src, tgt]["causal_strength"] = new_str
                    combined.edges[src, tgt]["edge_type"] = "both"
            else:
                combined.add_edge(src, tgt, **data)
        self.combined_graph = combined

    # ------------------------------------------------------------------
    # Build structural graph
    # ------------------------------------------------------------------

    def build_structural_graph(self) -> nx.DiGraph:
        """Construct the structural causal graph from topology knowledge.

        Encodes known causal relationships:
        - Link congestion → packet loss, latency
        - Link congestion → downstream node latency
        - Node CPU → temperature → buffer drops
        - Co-located links → traffic overflow (bidirectional)
        """
        g = nx.DiGraph()
        links = self._topology.get_all_links()

        # Track which links connect to each node (for co-location overflow edges)
        node_links: dict[str, list[str]] = defaultdict(list)
        for lnk in links:
            lid = lnk["link_id"]
            node_links[lnk["source"]].append(lid)
            node_links[lnk["target"]].append(lid)

        # Per-link structural edges
        for lnk in links:
            lid = lnk["link_id"]
            _, downstream = self._link_direction(lnk)

            util_var = _causal_var(lid, "utilization_pct")
            loss_var = _causal_var(lid, "packet_loss_pct")
            lat_var  = _causal_var(lid, "latency_ms")
            dn_lat   = f"{downstream}_latency"   # virtual node latency variable

            # Congestion causes packet loss and latency on same link
            self._add_edge(g, util_var, loss_var, 0.75)
            self._add_edge(g, util_var, lat_var,  0.65)

            # Congestion on link degrades downstream node's experienced latency
            self._add_edge(g, util_var, dn_lat, 0.60)

            # Packet loss causes downstream latency (TCP retransmits)
            self._add_edge(g, loss_var, dn_lat, 0.50)

        # Per-node structural edges
        for node_id in self._graph_topo.nodes:
            cpu_var  = _causal_var(node_id, "cpu_pct")
            temp_var = _causal_var(node_id, "temperature_c")
            drop_var = _causal_var(node_id, "buffer_drops")
            mem_var  = _causal_var(node_id, "memory_pct")

            self._add_edge(g, cpu_var,  temp_var, 0.70)   # CPU load heats device
            self._add_edge(g, temp_var, drop_var, 0.65)   # overheating causes drops
            self._add_edge(g, cpu_var,  drop_var, 0.55)   # high CPU → buffer drops
            self._add_edge(g, mem_var,  drop_var, 0.45)   # memory pressure → drops

        # Co-location: pairs of links at the same node → traffic overflow
        for node_id, link_ids in node_links.items():
            unique_ids = sorted(set(link_ids))
            for lid1, lid2 in combinations(unique_ids, 2):
                v1 = _causal_var(lid1, "utilization_pct")
                v2 = _causal_var(lid2, "utilization_pct")
                self._add_edge(g, v1, v2, 0.40)
                self._add_edge(g, v2, v1, 0.40)

        self.structural_graph = g
        self._merge_graphs()
        return g

    # ------------------------------------------------------------------
    # Data-driven causal discovery
    # ------------------------------------------------------------------

    def learn_from_data(self, telemetry_df: pd.DataFrame) -> None:
        """Discover causal edges from telemetry data.

        Tries causalnex NOTEARS first; falls back to lagged correlation analysis.
        Updates learned_graph and combined_graph.
        """
        metric_df = self._build_metric_df(telemetry_df)
        if metric_df.empty:
            print("  [CausalEngine] Empty metric DataFrame — skipping data learning.")
            return

        # Fill gaps and drop all-NaN columns
        metric_df = metric_df.ffill().bfill().dropna(axis=1, how="all")

        learned = nx.DiGraph()
        try:
            self._learn_notears(metric_df, learned)
            print(f"  [CausalEngine] NOTEARS learned {learned.number_of_edges()} edges.")
        except Exception as exc:
            print(f"  [CausalEngine] NOTEARS unavailable ({exc.__class__.__name__}: {exc}). "
                  "Falling back to lagged correlation.")
            self._learn_lagged_correlation(metric_df, learned)
            print(f"  [CausalEngine] Lagged correlation learned {learned.number_of_edges()} edges.")

        self.learned_graph = learned
        self._merge_graphs()

    def _build_metric_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot the flat telemetry DataFrame into wide format.

        Each column is "{entity_id}_{short_metric}".
        """
        if "entity_id" not in df.columns:
            return pd.DataFrame()

        ts_col = pd.to_datetime(df["timestamp"], utc=True)
        frames: list[pd.DataFrame] = []

        for entity_id, group in df.groupby("entity_id"):
            metric_cols = [
                c for c in group.columns
                if c not in ("timestamp", "entity_id", "entity_type")
                and c in _METRIC_SHORT
            ]
            if not metric_cols:
                continue
            sub = group.set_index(ts_col.loc[group.index])[metric_cols].copy()
            sub.columns = [_causal_var(str(entity_id), c) for c in metric_cols]
            frames.append(sub)

        if not frames:
            return pd.DataFrame()

        wide = pd.concat(frames, axis=1)
        wide = wide[~wide.index.duplicated(keep="first")].sort_index()
        return wide

    def _learn_notears(self, metric_df: pd.DataFrame, graph: nx.DiGraph) -> None:
        """Add edges from pgmpy HillClimbSearch structure learning (replaces causalnex NOTEARS)."""
        from pgmpy.estimators import HillClimbSearch, BIC  # type: ignore

        # Limit columns to avoid O(n^2) blowup; prefer nodes known to the structural graph
        known_vars = set(self.structural_graph.nodes)
        cols = [c for c in metric_df.columns if c in known_vars] or list(metric_df.columns)
        sub_df = metric_df[cols[:30]].astype(float)  # cap at 30 vars for tractability

        hc = HillClimbSearch(data=sub_df)
        dag = hc.estimate(scoring_method=BIC(data=sub_df), max_iter=200)
        for src, tgt in dag.edges():
            if src in self.structural_graph.nodes or tgt in self.structural_graph.nodes:
                self._add_edge(graph, str(src), str(tgt), 0.70, edge_type="learned")

    def _learn_lagged_correlation(
        self, metric_df: pd.DataFrame, graph: nx.DiGraph
    ) -> None:
        """Add edges based on lagged cross-correlation between metric series."""
        cols = list(metric_df.columns)
        arr = metric_df.values.astype(float)
        n = arr.shape[0]

        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                if i == j:
                    continue
                best_corr = 0.0
                best_lag = 0
                for lag in range(1, _MAX_LAG_STEPS + 1):
                    if lag >= n:
                        break
                    x = arr[:-lag, i]
                    y = arr[lag:, j]
                    mask = np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 10:
                        continue
                    corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag

                if best_corr >= _CORRELATION_THRESHOLD:
                    # col_i at t precedes col_j at t+lag → causal edge i→j
                    self._add_edge(graph, col_i, col_j, round(best_corr, 4),
                                   edge_type="learned")

    # ------------------------------------------------------------------
    # Root-cause analysis
    # ------------------------------------------------------------------

    def find_root_cause(self, anomalies: list[Anomaly]) -> list[Hypothesis]:
        """Walk backwards through the causal graph to hypothesize root causes.

        For each anomaly, finds upstream candidates and scores them by:
        - causal_strength of the connecting path
        - temporal precedence (anomaly with earlier timestamp → more likely root)
        - coverage (candidates explaining more anomalies rank higher)

        Returns hypotheses sorted by confidence descending.
        """
        if not anomalies:
            return []

        g = self.combined_graph

        # Map causal var → anomaly (most severe per var)
        var_to_anomaly: dict[str, Anomaly] = {}
        for a in anomalies:
            v = _anomaly_to_causal_var(a)
            if v is None:
                continue
            if v not in var_to_anomaly or _severity_rank(a.severity) > _severity_rank(
                var_to_anomaly[v].severity
            ):
                var_to_anomaly[v] = a

        observed_vars = set(var_to_anomaly)

        # Collect candidate root-cause nodes by BFS up to 4 hops upstream
        candidate_scores: dict[str, dict[str, Any]] = {}

        for obs_var, obs_anomaly in var_to_anomaly.items():
            if obs_var not in g:
                continue
            for candidate, path in _bfs_predecessors(g, obs_var, max_depth=4):
                if candidate in observed_vars:
                    # Observed anomaly — not a candidate root cause (it's already explained)
                    continue
                path_strength = _path_strength(g, path)
                temporal_score = _temporal_score(
                    anomalies, candidate, obs_anomaly.timestamp
                )

                if candidate not in candidate_scores:
                    candidate_scores[candidate] = {
                        "total_strength": 0.0,
                        "count": 0,
                        "explained": [],
                        "temporal_scores": [],
                    }

                rec = candidate_scores[candidate]
                rec["total_strength"] += path_strength
                rec["count"] += 1
                rec["explained"].append(obs_var)
                rec["temporal_scores"].append(temporal_score)

        if not candidate_scores:
            # No upstream causes found — report the most severe anomaly itself
            return self._self_explain(anomalies)

        # Score candidates
        scored: list[tuple[float, str, dict]] = []
        total_obs = len(observed_vars)
        for cand, rec in candidate_scores.items():
            avg_strength = rec["total_strength"] / max(rec["count"], 1)
            avg_temporal = (
                sum(rec["temporal_scores"]) / len(rec["temporal_scores"])
                if rec["temporal_scores"] else 0.5
            )
            coverage = len(set(rec["explained"])) / max(total_obs, 1)
            score = (avg_strength * 0.45) + (avg_temporal * 0.25) + (coverage * 0.30)
            scored.append((score, cand, rec))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Build hypotheses — group by root cause if it explains ≥2 anomalies
        hypotheses: list[Hypothesis] = []
        seen_roots: set[str] = set()

        for score, cand_var, rec in scored[:5]:  # top 5 candidates
            if cand_var in seen_roots:
                continue
            seen_roots.add(cand_var)

            explained_anomalies = [var_to_anomaly[v] for v in rec["explained"] if v in var_to_anomaly]
            affected_nodes = sorted(set(
                a.node_id for a in explained_anomalies if a.node_id
            ))
            affected_links = sorted(set(
                a.link_id for a in explained_anomalies if a.link_id
            ))
            evidence = [
                f"{a.metric_name}={a.observed_value:.2f} (expected≈{a.expected_value:.2f}) "
                f"on {'link ' + a.link_id if a.link_id else 'node ' + str(a.node_id)}"
                for a in explained_anomalies
            ]

            entity_id, metric = _split_causal_var(cand_var)
            description = (
                f"Elevated {metric} on {entity_id} is likely driving "
                f"{len(explained_anomalies)} downstream anomalies."
            )

            hypotheses.append(Hypothesis(
                description=description,
                root_cause=cand_var,
                confidence=round(min(score, 0.99), 4),
                evidence=evidence,
                affected_nodes=affected_nodes,
                affected_links=affected_links,
                recommended_actions=[
                    f"Investigate {entity_id} {metric}",
                    f"Check upstream traffic on {entity_id}",
                ],
            ))

        if not hypotheses:
            hypotheses = self._self_explain(anomalies)

        return hypotheses

    def _self_explain(self, anomalies: list[Anomaly]) -> list[Hypothesis]:
        """Fallback: return a hypothesis for the most severe anomaly itself."""
        ranked = sorted(anomalies, key=lambda a: _severity_rank(a.severity), reverse=True)
        best = ranked[0]
        entity = best.link_id or best.node_id or "unknown"
        return [Hypothesis(
            description=f"No upstream cause found. {best.metric_name} anomaly on {entity} may be the root cause.",
            root_cause=_causal_var(entity, best.metric_name),
            confidence=round(best.confidence * 0.6, 4),
            evidence=[f"{best.metric_name}={best.observed_value:.2f} on {entity}"],
            affected_nodes=[best.node_id] if best.node_id else [],
            affected_links=[best.link_id] if best.link_id else [],
            recommended_actions=[f"Inspect {entity} directly"],
        )]

    # ------------------------------------------------------------------
    # Counterfactual reasoning
    # ------------------------------------------------------------------

    def run_counterfactual(
        self, action: dict, current_state: dict[str, dict[str, float]]
    ) -> dict:
        """Simulate the downstream effect of taking an action.

        Args:
            action: dict with keys: action_type ("reroute"|"rate_limit"),
                    plus action-specific parameters.
            current_state: {entity_id: {metric_name: value}}

        Returns:
            {
              "predicted_state": {entity_id: {metric_name: new_value}},
              "affected_entities": [entity_id, ...],
              "risk_score": fraction of entities where a metric worsened,
            }
        """
        action_type = action.get("action_type", "")
        if action_type == "reroute":
            return self._counterfactual_reroute(action, current_state)
        if action_type == "rate_limit":
            return self._counterfactual_rate_limit(action, current_state)
        return {
            "predicted_state": {},
            "affected_entities": [],
            "risk_score": 0.0,
            "error": f"Unknown action_type: {action_type}",
        }

    def _counterfactual_reroute(
        self, action: dict, current_state: dict[str, dict[str, float]]
    ) -> dict:
        """Reroute: traffic moves from from_link to to_link."""
        from_link = action.get("from_link", "")
        to_link   = action.get("to_link", "")
        fraction  = float(action.get("traffic_fraction", 0.5))

        # Delta utilization
        from_util = current_state.get(from_link, {}).get("utilization_pct", 0.0)
        to_util   = current_state.get(to_link,   {}).get("utilization_pct", 0.0)
        traffic_delta = from_util * fraction

        deltas: dict[str, dict[str, float]] = {
            from_link: {"utilization_pct": -traffic_delta},
            to_link:   {"utilization_pct": +traffic_delta},
        }

        return self._propagate_deltas(deltas, current_state)

    def _counterfactual_rate_limit(
        self, action: dict, current_state: dict[str, dict[str, float]]
    ) -> dict:
        """Rate-limit: reduce utilization on target_link by reduction_pct."""
        target = action.get("target_link", "")
        reduction = float(action.get("reduction_pct", 20.0))
        current_util = current_state.get(target, {}).get("utilization_pct", 0.0)
        actual_reduction = min(reduction, current_util)

        deltas: dict[str, dict[str, float]] = {
            target: {"utilization_pct": -actual_reduction},
        }
        return self._propagate_deltas(deltas, current_state)

    def _propagate_deltas(
        self,
        initial_deltas: dict[str, dict[str, float]],
        current_state: dict[str, dict[str, float]],
    ) -> dict:
        """Walk forward through the STRUCTURAL causal graph, propagating metric deltas.

        Uses only the structural graph (not the dense learned graph) to avoid
        feedback cycles from correlation noise amplifying deltas.

        For each downstream variable V:
            delta(V) += sum(causal_strength(U→V) * delta(U) for all upstream U)
        """
        g = self.structural_graph  # intentionally NOT combined_graph

        # Convert entity-level deltas to causal-var deltas
        var_deltas: dict[str, float] = {}
        for entity_id, metrics in initial_deltas.items():
            for metric_name, delta in metrics.items():
                var_deltas[_causal_var(entity_id, metric_name)] = delta

        # BFS forward propagation (up to 4 hops, no revisiting)
        visited: set[str] = set()
        frontier = list(var_deltas.keys())

        for _ in range(4):
            next_frontier: list[str] = []
            for src_var in frontier:
                if src_var in visited:
                    continue
                visited.add(src_var)
                src_delta = var_deltas.get(src_var, 0.0)
                if src_var not in g:
                    continue
                for tgt_var in g.successors(src_var):
                    strength = g.edges[src_var, tgt_var].get("causal_strength", 0.5)
                    propagated = src_delta * strength
                    var_deltas[tgt_var] = var_deltas.get(tgt_var, 0.0) + propagated
                    next_frontier.append(tgt_var)
            frontier = next_frontier

        # Reconstruct predicted_state only for entities present in current_state
        predicted_state: dict[str, dict[str, float]] = {}
        worsened = 0
        total = 0

        for var, delta in var_deltas.items():
            if abs(delta) < 0.01:
                continue
            entity_id, metric_short = _split_causal_var(var)
            if entity_id not in current_state:
                continue
            entity_current = current_state[entity_id]
            metric_full = _find_full_metric(metric_short, entity_current)
            if metric_full is None:
                continue  # entity known but metric not tracked in current_state

            current_val = entity_current[metric_full]
            new_val = max(0.0, current_val + delta)

            predicted_state.setdefault(entity_id, {})[metric_full] = round(new_val, 4)
            total += 1
            if delta > 0:
                worsened += 1

        risk_score = round(worsened / max(total, 1), 4)
        affected_entities = sorted(predicted_state)

        return {
            "predicted_state": predicted_state,
            "affected_entities": affected_entities,
            "risk_score": risk_score,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(
        self,
        filename: str = "causal_graph.png",
        anomalous_vars: Optional[set[str]] = None,
        root_cause_vars: Optional[set[str]] = None,
    ) -> None:
        """Draw the causal graph.

        Edge widths are proportional to causal_strength.
        Colors: anomalous=red, root_cause=orange, normal=green.
        """
        g = self.combined_graph
        if g.number_of_nodes() == 0:
            print("  [CausalEngine] Graph is empty — nothing to visualize.")
            return

        anomalous_vars = anomalous_vars or set()
        root_cause_vars = root_cause_vars or set()

        node_colors = []
        for node in g.nodes:
            if node in root_cause_vars:
                node_colors.append(_NODE_VIZ_COLORS["root_cause"])
            elif node in anomalous_vars:
                node_colors.append(_NODE_VIZ_COLORS["anomalous"])
            else:
                node_colors.append(_NODE_VIZ_COLORS["normal"])

        edge_widths = [
            max(0.5, g.edges[u, v].get("causal_strength", 0.5) * 4)
            for u, v in g.edges
        ]
        edge_colors = [
            "#c0392b" if g.edges[u, v].get("edge_type") == "learned"
            else "#2980b9" if g.edges[u, v].get("edge_type") == "both"
            else "#7f8c8d"
            for u, v in g.edges
        ]

        fig, ax = plt.subplots(figsize=(22, 14))
        pos = nx.spring_layout(g, k=2.5, seed=42)

        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors,
                               node_size=300, alpha=0.85)
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=5)
        nx.draw_networkx_edges(g, pos, ax=ax, width=edge_widths,
                               edge_color=edge_colors, alpha=0.7,
                               arrows=True, arrowsize=10,
                               connectionstyle="arc3,rad=0.1")

        import matplotlib.patches as mpatches
        legend = [
            mpatches.Patch(color=_NODE_VIZ_COLORS["root_cause"], label="Root cause candidate"),
            mpatches.Patch(color=_NODE_VIZ_COLORS["anomalous"],  label="Anomalous"),
            mpatches.Patch(color=_NODE_VIZ_COLORS["normal"],     label="Normal"),
            mpatches.Patch(color="#7f8c8d", label="Structural edge"),
            mpatches.Patch(color="#c0392b", label="Learned edge"),
            mpatches.Patch(color="#2980b9", label="Both"),
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=7)
        ax.set_title("CyberCypher 5.0 — Causal Graph (Digital Twin)", fontsize=13, fontweight="bold")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [CausalEngine] Causal graph saved to {filename}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_graph_summary(self) -> dict:
        """Return summary statistics about the combined causal graph."""
        g = self.combined_graph
        edges_with_strength = [
            (u, v, g.edges[u, v].get("causal_strength", 0.0))
            for u, v in g.edges
        ]
        strongest = sorted(edges_with_strength, key=lambda x: x[2], reverse=True)[:10]
        return {
            "num_nodes": g.number_of_nodes(),
            "num_edges": g.number_of_edges(),
            "structural_edges": self.structural_graph.number_of_edges(),
            "learned_edges": self.learned_graph.number_of_edges(),
            "strongest_edges": [
                {"from": u, "to": v, "causal_strength": round(s, 4)}
                for u, v, s in strongest
            ],
        }


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _severity_rank(severity: str) -> int:
    return {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(severity, 0)


def _path_strength(g: nx.DiGraph, path: list[str]) -> float:
    """Compute the product of causal strengths along a path (decays with hops)."""
    strength = 1.0
    for i in range(len(path) - 1):
        edge_str = g.edges[path[i], path[i + 1]].get("causal_strength", 0.5)
        strength *= edge_str
    return strength


def _bfs_predecessors(
    g: nx.DiGraph, start: str, max_depth: int = 4
) -> list[tuple[str, list[str]]]:
    """BFS from start BACKWARDS; yields (predecessor_node, path_from_pred_to_start)."""
    results: list[tuple[str, list[str]]] = []
    frontier = [(start, [start])]
    visited = {start}

    for _ in range(max_depth):
        next_frontier = []
        for node, path in frontier:
            for pred in g.predecessors(node):
                if pred in visited:
                    continue
                visited.add(pred)
                new_path = [pred] + path
                results.append((pred, new_path))
                next_frontier.append((pred, new_path))
        frontier = next_frontier
        if not frontier:
            break

    return results


def _temporal_score(anomalies: list[Anomaly], candidate_var: str, obs_ts: Any) -> float:
    """Score 0-1: does this candidate appear before the observed anomaly?

    If no anomaly is found for the candidate var, default to 0.5 (neutral).
    """
    for a in anomalies:
        if _anomaly_to_causal_var(a) == candidate_var:
            if a.timestamp < obs_ts:
                return 0.9   # candidate preceded the observation → strong support
            if a.timestamp == obs_ts:
                return 0.5
            return 0.2      # candidate came AFTER → unlikely root cause
    return 0.5  # unknown timing


def _split_causal_var(var: str) -> tuple[str, str]:
    """Split 'AGG1-CR1_utilization' → ('AGG1-CR1', 'utilization').

    Handles link IDs with hyphens by using the LAST underscore as delimiter.
    """
    idx = var.rfind("_")
    if idx == -1:
        return var, ""
    return var[:idx], var[idx + 1:]


def _find_full_metric(short: str, entity_metrics: dict[str, float]) -> Optional[str]:
    """Reverse-lookup: find the full metric column name for a short suffix."""
    for full, s in _METRIC_SHORT.items():
        if s == short and full in entity_metrics:
            return full
    return None


# ---------------------------------------------------------------------------
# Main — demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    from datetime import datetime, timezone

    import numpy as np

    from src.simulator.anomaly_injector import AnomalyInjector
    from src.simulator.telemetry import TelemetryGenerator

    rng = np.random.default_rng(42)

    # ── 1. Build topology + causal engine ───────────────────────────────────
    print("\n── Phase 1: Build topology & structural causal graph ─────────────")
    topo = NetworkTopology()
    engine = CausalEngine(topo)
    summary = engine.get_graph_summary()
    print(f"  Structural graph : {summary['structural_edges']} edges")
    print(f"  Combined graph   : {summary['num_nodes']} nodes, {summary['num_edges']} edges")
    print(f"  Top 3 causal edges:")
    for e in summary["strongest_edges"][:3]:
        print(f"    {e['from']:35s} → {e['to']:35s}  strength={e['causal_strength']:.2f}")

    # ── 2. Generate 24h telemetry + inject congestion cascade ───────────────
    print("\n── Phase 2: Generate 24h telemetry + inject anomaly ──────────────")
    gen      = TelemetryGenerator(topo, seed=42)
    baseline = gen.generate_baseline(duration_hours=24)
    injector = AnomalyInjector(topo, seed=7)
    anomalous_df, labels = injector.inject_random_scenarios(
        baseline, n_scenarios=2, min_gap_minutes=60
    )
    print(f"  Baseline rows   : {len(baseline):,}")
    print(f"  Injected labels : {[l.scenario_type for l in labels]}")

    # ── 3. Learn from data ──────────────────────────────────────────────────
    print("\n── Phase 3: Learn causal edges from telemetry ────────────────────")
    engine.learn_from_data(anomalous_df)
    summary = engine.get_graph_summary()
    print(f"  Learned edges   : {summary['learned_edges']}")
    print(f"  Combined edges  : {summary['num_edges']}")

    # ── 4. Synthesize anomalies for root-cause analysis ─────────────────────
    print("\n── Phase 4: Root-cause analysis ──────────────────────────────────")
    now = datetime.now(timezone.utc)
    # Simulate: CR1-CR2 link is congested → downstream AGG1-CR1 also congested → EDGE1 drops
    synthetic_anomalies = [
        Anomaly(
            timestamp=now,
            metric_name="utilization_pct",
            link_id="CR1-CR2",
            observed_value=95.0,
            expected_value=55.0,
            severity="critical",
            confidence=0.92,
            detector_type="threshold",
        ),
        Anomaly(
            timestamp=now,
            metric_name="packet_loss_pct",
            link_id="AGG1-CR1",
            observed_value=8.5,
            expected_value=0.5,
            severity="high",
            confidence=0.85,
            detector_type="ewma",
        ),
        Anomaly(
            timestamp=now,
            metric_name="buffer_drops",
            node_id="EDGE1",
            observed_value=4500.0,
            expected_value=300.0,
            severity="high",
            confidence=0.78,
            detector_type="isolation_forest",
        ),
    ]

    hypotheses = engine.find_root_cause(synthetic_anomalies)
    print(f"  Input anomalies : {len(synthetic_anomalies)}")
    print(f"  Hypotheses found: {len(hypotheses)}")
    for i, h in enumerate(hypotheses, 1):
        print(f"\n  Hypothesis #{i}:")
        print(f"    root_cause  : {h.root_cause}")
        print(f"    confidence  : {h.confidence:.4f}")
        print(f"    description : {h.description}")
        print(f"    evidence    ({len(h.evidence)}):")
        for ev in h.evidence[:3]:
            print(f"      • {ev}")
        print(f"    affected_links : {h.affected_links}")
        print(f"    affected_nodes : {h.affected_nodes}")

    # ── 5. Counterfactual: reroute traffic from congested link ─────────────
    print("\n── Phase 5: Counterfactual reasoning ─────────────────────────────")
    current_state = {
        "CR1-CR2":  {"utilization_pct": 95.0, "packet_loss_pct": 3.2, "latency_ms": 18.0},
        "AGG1-CR1": {"utilization_pct": 72.0, "packet_loss_pct": 8.5, "latency_ms": 12.0},
        "AGG2-CR1": {"utilization_pct": 35.0, "packet_loss_pct": 0.2, "latency_ms":  4.0},
        "CR1":      {"cpu_pct": 88.0, "temperature_c": 68.0, "buffer_drops": 1200},
        "EDGE1":    {"cpu_pct": 75.0, "temperature_c": 52.0, "buffer_drops": 4500},
    }

    reroute_action = {
        "action_type":      "reroute",
        "from_link":        "CR1-CR2",
        "to_link":          "AGG2-CR1",
        "traffic_fraction": 0.40,
    }

    print(f"  Action: reroute 40% of traffic from CR1-CR2 → AGG2-CR1")
    cf_result = engine.run_counterfactual(reroute_action, current_state)
    print(f"  Affected entities : {cf_result['affected_entities']}")
    print(f"  Risk score        : {cf_result['risk_score']:.4f}  "
          f"({'some metrics worsen' if cf_result['risk_score'] > 0.3 else 'mostly beneficial'})")
    print(f"  Predicted state changes:")
    for entity_id, metrics in sorted(cf_result["predicted_state"].items()):
        orig = current_state.get(entity_id, {})
        for metric, new_val in sorted(metrics.items()):
            old_val = orig.get(metric, 0.0)
            delta   = new_val - old_val
            arrow   = "▲" if delta > 0 else "▼"
            print(f"    {entity_id:12s}  {metric:20s}  {old_val:.2f} → {new_val:.2f}  {arrow}{abs(delta):.2f}")

    # Rate-limit counterfactual
    print()
    rate_action = {
        "action_type":  "rate_limit",
        "target_link":  "CR1-CR2",
        "reduction_pct": 30.0,
    }
    print(f"  Action: rate-limit CR1-CR2 by 30%")
    rl_result = engine.run_counterfactual(rate_action, current_state)
    print(f"  Affected entities : {rl_result['affected_entities']}")
    print(f"  Risk score        : {rl_result['risk_score']:.4f}")
    print(f"  Predicted state changes:")
    for entity_id, metrics in sorted(rl_result["predicted_state"].items()):
        orig = current_state.get(entity_id, {})
        for metric, new_val in sorted(metrics.items()):
            old_val = orig.get(metric, 0.0)
            delta   = new_val - old_val
            arrow   = "▲" if delta > 0 else "▼"
            print(f"    {entity_id:12s}  {metric:20s}  {old_val:.2f} → {new_val:.2f}  {arrow}{abs(delta):.2f}")

    # ── 6. Visualize ────────────────────────────────────────────────────────
    print("\n── Phase 6: Visualize causal graph ───────────────────────────────")
    anomalous_vars = {_anomaly_to_causal_var(a) for a in synthetic_anomalies} - {None}  # type: ignore[operator]
    root_cause_vars = {h.root_cause for h in hypotheses}
    engine.visualize(
        "causal_graph.png",
        anomalous_vars=anomalous_vars,
        root_cause_vars=root_cause_vars,
    )

    print("\n🎉 CausalEngine demo complete.\n")
