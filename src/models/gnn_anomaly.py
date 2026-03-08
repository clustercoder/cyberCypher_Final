"""Graph Neural Network anomaly detector for network topology-aware detection.

Uses torch-geometric GCNConv layers to learn anomaly patterns propagated across
the network graph. Falls back gracefully if torch-geometric is not installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv
    _PYGEOMETRIC_AVAILABLE = True
except ImportError:
    _PYGEOMETRIC_AVAILABLE = False
    GCNConv = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from src.simulator.topology import NetworkTopology

# 4 node features: latency_ms, packet_loss_pct, utilization_pct, buffer_drops
_NODE_FEATURE_KEYS = ("latency_ms", "packet_loss_pct", "utilization_pct", "buffer_drops")
_NUM_NODE_FEATURES = len(_NODE_FEATURE_KEYS)


class _NetworkGNNNet(nn.Module):
    """Two-layer GCN for node-level anomaly scoring."""

    def __init__(self, in_channels: int = _NUM_NODE_FEATURES, hidden_channels: int = 32) -> None:
        super().__init__()
        if not _PYGEOMETRIC_AVAILABLE:
            raise RuntimeError("torch-geometric is required for GNN anomaly detection.")
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass returning per-node anomaly probability in [0, 1]."""
        x = self.relu(self.conv1(x, edge_index))
        x = self.sigmoid(self.conv2(x, edge_index))
        return x.squeeze(-1)


class GNNAnomalyDetector:
    """Topology-aware GNN anomaly detector.

    Encodes the ISP network as a graph, builds node feature matrices from
    per-node metrics, and scores each node with a learned GCN. Falls back
    to threshold-based scoring when torch-geometric is unavailable.

    Parameters
    ----------
    topology:
        NetworkTopology instance used to build the edge index.
    hidden_channels:
        Number of hidden units in each GCN layer.
    anomaly_threshold:
        Score above which a node is considered anomalous (0-1).
    """

    def __init__(
        self,
        topology: NetworkTopology,
        hidden_channels: int = 32,
        anomaly_threshold: float = 0.6,
    ) -> None:
        self.topology = topology
        self.anomaly_threshold = anomaly_threshold
        self._available = _PYGEOMETRIC_AVAILABLE
        self._node_ids: list[str] = [n["node_id"] for n in topology.get_all_nodes()]
        self._node_index: dict[str, int] = {nid: i for i, nid in enumerate(self._node_ids)}
        self._edge_index: torch.Tensor | None = self._build_edge_index()
        self._model: _NetworkGNNNet | None = None
        self._trained: bool = False

        if self._available:
            self._model = _NetworkGNNNet(
                in_channels=_NUM_NODE_FEATURES,
                hidden_channels=hidden_channels,
            )
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
            self._criterion = nn.BCELoss()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        node_features_seq: list[dict[str, dict[str, float]]],
        labels_seq: list[dict[str, int]],
        epochs: int = 30,
    ) -> list[float]:
        """Train the GCN on a sequence of snapshot–label pairs.

        Parameters
        ----------
        node_features_seq:
            List of {node_id: {metric: value}} dicts (one per time step).
        labels_seq:
            List of {node_id: 0|1} binary anomaly labels (same length).
        epochs:
            Training epochs over the full sequence.

        Returns
        -------
        Per-epoch average loss history.
        """
        if not self._available or self._model is None:
            return []

        self._model.train()
        loss_history: list[float] = []

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            steps = 0
            for snapshot, labels in zip(node_features_seq, labels_seq):
                x = self._build_feature_tensor(snapshot)
                y = self._build_label_tensor(labels)
                if self._edge_index is None:
                    continue

                self._optimizer.zero_grad()
                pred = self._model(x, self._edge_index)
                loss = self._criterion(pred, y)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
                steps += 1

            avg = epoch_loss / max(steps, 1)
            loss_history.append(avg)

        self._model.eval()
        self._trained = True
        return loss_history

    def score(self, node_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
        """Return per-node anomaly scores in [0, 1].

        Parameters
        ----------
        node_metrics:
            {node_id: {metric_name: value}} for the current telemetry snapshot.
        """
        if not self._available or self._model is None or self._edge_index is None:
            return self._fallback_score(node_metrics)

        x = self._build_feature_tensor(node_metrics)
        with torch.no_grad():
            scores = self._model(x, self._edge_index).cpu().numpy()

        return {nid: float(scores[i]) for i, nid in enumerate(self._node_ids)}

    def detect(self, node_metrics: dict[str, dict[str, float]]) -> list[str]:
        """Return list of node IDs whose anomaly score exceeds the threshold."""
        scores = self.score(node_metrics)
        return [nid for nid, score in scores.items() if score >= self.anomaly_threshold]

    def is_available(self) -> bool:
        """Return True if torch-geometric is installed and model is usable."""
        return self._available

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_edge_index(self) -> torch.Tensor | None:
        """Build COO-format edge_index tensor from the NetworkX topology graph."""
        graph: nx.Graph = self.topology.get_graph()
        src_list: list[int] = []
        dst_list: list[int] = []

        for u, v in graph.edges():
            u_str, v_str = str(u), str(v)
            if u_str not in self._node_index or v_str not in self._node_index:
                continue
            src_list.append(self._node_index[u_str])
            dst_list.append(self._node_index[v_str])
            # Undirected: add both directions
            src_list.append(self._node_index[v_str])
            dst_list.append(self._node_index[u_str])

        if not src_list:
            return None

        return torch.tensor([src_list, dst_list], dtype=torch.long)

    def _build_feature_tensor(self, node_metrics: dict[str, dict[str, float]]) -> torch.Tensor:
        """Build [num_nodes, 4] float32 feature tensor from metric snapshot."""
        rows: list[list[float]] = []
        for nid in self._node_ids:
            metrics = node_metrics.get(nid, {})
            row = [float(metrics.get(key, 0.0)) for key in _NODE_FEATURE_KEYS]
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32)

    def _build_label_tensor(self, labels: dict[str, int]) -> torch.Tensor:
        """Build [num_nodes] float32 label tensor (0 or 1)."""
        vals = [float(labels.get(nid, 0)) for nid in self._node_ids]
        return torch.tensor(vals, dtype=torch.float32)

    def _fallback_score(self, node_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
        """Simple threshold-based fallback when torch-geometric is unavailable."""
        scores: dict[str, float] = {}
        thresholds = {
            "utilization_pct": 85.0,
            "latency_ms": 50.0,
            "packet_loss_pct": 1.0,
            "buffer_drops": 100.0,
        }
        for nid in self._node_ids:
            metrics = node_metrics.get(nid, {})
            violations = 0
            for metric, thresh in thresholds.items():
                val = metrics.get(metric, 0.0)
                if val > thresh:
                    violations += 1
            scores[nid] = min(1.0, violations / len(thresholds))
        return scores
