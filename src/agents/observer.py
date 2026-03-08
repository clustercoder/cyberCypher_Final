from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.models.anomaly_detection import (
    AnomalyDetectorEnsemble,
    EWMADetector,
    IsolationForestDetector,
    ThresholdDetector,
)
from src.models.forecasting import FallbackForecaster, LSTMForecaster
from src.models.schemas import Anomaly
from src.simulator.topology import NetworkTopology
from src.utils.logger import logger

try:
    from src.models.gnn_anomaly import GNNAnomalyDetector
    _GNN_AVAILABLE = True
except ImportError:
    _GNN_AVAILABLE = False

Severity = Literal["low", "medium", "high", "critical"]

_SEVERITY_ORDER: dict[Severity, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

_NODE_METRICS: tuple[str, ...] = (
    "cpu_pct",
    "memory_pct",
    "temperature_c",
    "buffer_drops",
)
_LINK_METRICS: tuple[str, ...] = (
    "utilization_pct",
    "latency_ms",
    "packet_loss_pct",
    "throughput_gbps",
)
_DYNAMIC_METRICS: tuple[str, ...] = _NODE_METRICS + _LINK_METRICS

_FORECAST_LINK_METRICS: tuple[str, ...] = ("utilization_pct",)
_FORECAST_NODE_METRICS: tuple[str, ...] = ("cpu_pct", "memory_pct", "temperature_c")
_FORECAST_THRESHOLDS: dict[str, float] = {
    "utilization_pct": 85.0,
    "cpu_pct": 90.0,
    "memory_pct": 85.0,
    "temperature_c": 70.0,
}

_REQUIRED_BASELINE_COLUMNS: set[str] = {"timestamp", "entity_id", "entity_type"}


class ObserverAgent:
    """Telemetry observer that performs ensemble detection and proactive forecasting."""

    def __init__(
        self,
        topology: NetworkTopology,
        baselines: dict[str, dict[str, float]],
    ) -> None:
        """Initialize detector ensemble, forecasting state, and anomaly memory."""
        self.topology = topology
        self.baselines = baselines

        latency_baselines = self._build_latency_baselines()
        self.threshold_detector = ThresholdDetector(latency_baselines=latency_baselines)
        self.ewma_detect = EWMADetector(span=20)
        self.ewma_ingest = EWMADetector(span=20)
        self.isolation_forest_detector = IsolationForestDetector(contamination=0.05)

        self.ensemble = AnomalyDetectorEnsemble(
            [self.threshold_detector, self.ewma_detect, self.isolation_forest_detector]
        )

        # Optional GNN anomaly detector (requires torch-geometric).
        self.gnn_detector: GNNAnomalyDetector | None = None
        if _GNN_AVAILABLE:
            try:
                self.gnn_detector = GNNAnomalyDetector(topology=topology)
                logger.info("GNNAnomalyDetector initialized successfully.")
            except Exception as exc:
                logger.warning("GNNAnomalyDetector init failed, skipping. error={}", exc)
                self.gnn_detector = None

        # 60 x 1-minute snapshots.
        self.history: deque[dict[str, Any]] = deque(maxlen=60)
        self.forecaster: LSTMForecaster | FallbackForecaster = FallbackForecaster()

        # key = "{entity}_{metric}".
        self._series_forecasters: dict[str, LSTMForecaster | FallbackForecaster] = {}
        self._series_metadata: dict[str, tuple[Literal["node", "link"], str, str]] = {}

        self.active_anomalies: dict[str, Anomaly] = {}
        self._anomaly_history: list[Anomaly] = []
        # Max MC dropout uncertainty score from the most recent detect() call.
        self._last_uncertainty_score: float = 0.0

    def train_detectors(self, baseline_df: pd.DataFrame) -> None:
        """Train ensemble components and initialize forecasting models."""
        self._validate_baseline_df(baseline_df)
        if baseline_df.empty:
            raise ValueError("baseline_df must not be empty")

        df = baseline_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")

        dynamic_numeric_cols = [
            metric
            for metric in _DYNAMIC_METRICS
            if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric])
        ]
        if not dynamic_numeric_cols:
            raise ValueError("baseline_df does not contain dynamic numeric metric columns")

        # Train IF only on dynamic metrics (not static metadata like capacity/baseline latency).
        self.isolation_forest_detector.train(df[dynamic_numeric_cols])

        self._warm_ewma_detectors(df)
        self._init_series_forecasters(df)
        self._train_gnn(df)

    def ingest(self, telemetry: dict[str, Any]) -> None:
        """Ingest one timestep from simulation engine output."""
        snapshot = self._sanitize_telemetry(telemetry)
        self.history.append(snapshot)

        for node_id, metrics in snapshot["nodes"].items():
            for metric_name, value in metrics.items():
                self.ewma_ingest.update(self._ewma_key(node_id, metric_name), value)

        for link_id, metrics in snapshot["links"].items():
            for metric_name, value in metrics.items():
                self.ewma_ingest.update(self._ewma_key(link_id, metric_name), value)

    def detect(self) -> list[Anomaly]:
        """Run ensemble + GNN + forecast detection and return new or materially updated anomalies."""
        if not self.history:
            return []

        self._last_uncertainty_score = 0.0
        latest = self.history[-1]
        observed = self.ensemble.detect_all(
            {
                "nodes": latest["nodes"],
                "links": latest["links"],
            }
        )

        gnn_anomalies = self._detect_gnn_anomalies(latest)
        predicted = self._detect_predicted_anomalies()

        all_observed = observed + gnn_anomalies
        cycle_anomalies = self._merge_cycle_anomalies(all_observed, predicted)
        emitted = self._apply_cycle_to_active_state(cycle_anomalies)
        return emitted

    def _detect_gnn_anomalies(self, snapshot: dict[str, Any]) -> list[Anomaly]:
        """Run GNN topology-aware anomaly scoring if detector is available."""
        if self.gnn_detector is None or not self.gnn_detector.is_available():
            return []

        try:
            # Build node_metrics dict: {node_id: {metric: value}}
            node_metrics: dict[str, dict[str, float]] = {}
            for node_id, metrics in snapshot["nodes"].items():
                node_metrics[str(node_id)] = {k: float(v) for k, v in metrics.items()}
            # Include link-adjacent metrics as pseudo-nodes if no node data
            if not node_metrics:
                return []

            scores = self.gnn_detector.score(node_metrics)
            anomalous_ids = self.gnn_detector.detect(node_metrics)
            anomalous_set = set(anomalous_ids)
            timestamp = self._parse_timestamp(snapshot["timestamp"])
            anomalies: list[Anomaly] = []
            for node_id, score in scores.items():
                if node_id not in anomalous_set:
                    continue
                severity = self._severity_from_ratio(score * 1.5)
                anomalies.append(
                    Anomaly(
                        timestamp=timestamp,
                        metric_name="gnn_anomaly_score",
                        node_id=node_id,
                        link_id=None,
                        observed_value=round(score, 6),
                        expected_value=0.5,
                        severity=severity,
                        confidence=round(min(score, 0.95), 4),
                        detector_type="gnn_topology",
                    )
                )
            return anomalies
        except Exception as exc:
            logger.warning("GNN detection failed: {}", exc)
            return []

    def get_forecast_uncertainty_score(self) -> float:
        """Return the max MC dropout uncertainty score from the most recent detect() call."""
        return self._last_uncertainty_score

    def get_active_anomalies(self) -> list[Anomaly]:
        """Return currently active anomalies."""
        anomalies = sorted(
            self.active_anomalies.values(),
            key=lambda a: (_SEVERITY_ORDER[a.severity], a.confidence),
            reverse=True,
        )
        return [a.model_copy(deep=True) for a in anomalies]

    def get_anomaly_history(self) -> list[Anomaly]:
        """Return all emitted anomalies (new + updates) since startup."""
        return [a.model_copy(deep=True) for a in self._anomaly_history]

    def get_network_health_summary(self) -> dict[str, object]:
        """Summarize active anomaly posture for control-plane/UI consumption."""
        active = list(self.active_anomalies.values())
        by_severity: dict[str, int] = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
        }
        for anomaly in active:
            by_severity[anomaly.severity] += 1

        worst_entity: str | None = None
        if active:
            worst = max(
                active,
                key=lambda a: (_SEVERITY_ORDER[a.severity], a.confidence),
            )
            worst_entity = worst.node_id or worst.link_id

        total = len(active)
        if by_severity["critical"] > 0:
            overall_health = "critical"
        elif by_severity["high"] > 0 or by_severity["medium"] > 0 or total >= 3:
            overall_health = "degraded"
        else:
            overall_health = "healthy"

        return {
            "total_anomalies": total,
            "by_severity": by_severity,
            "worst_entity": worst_entity,
            "overall_health": overall_health,
        }

    def _train_gnn(self, baseline_df: pd.DataFrame) -> None:
        """Train GNN anomaly detector using pseudo-labels derived from threshold rules."""
        if self.gnn_detector is None or not self.gnn_detector.is_available():
            return

        node_df = baseline_df[baseline_df["entity_type"] == "node"]
        if node_df.empty:
            logger.warning("No node rows in baseline_df; skipping GNN training.")
            return

        node_feature_seqs: list[dict[str, dict[str, float]]] = []
        label_seqs: list[dict[str, int]] = []

        timestamps = sorted(node_df["timestamp"].unique())
        for ts in timestamps:
            ts_df = node_df[node_df["timestamp"] == ts]
            snapshot: dict[str, dict[str, float]] = {}
            labels: dict[str, int] = {}
            for row in ts_df.itertuples(index=False):
                node_id = str(getattr(row, "entity_id"))
                metrics: dict[str, float] = {}
                for col in ("cpu_pct", "memory_pct", "temperature_c", "buffer_drops",
                            "latency_ms", "packet_loss_pct", "utilization_pct"):
                    val = getattr(row, col, None)
                    if val is not None and not pd.isna(val):
                        metrics[col] = float(val)
                snapshot[node_id] = metrics
                is_anomaly = (
                    metrics.get("cpu_pct", 0.0) > 85.0
                    or metrics.get("memory_pct", 0.0) > 80.0
                    or metrics.get("temperature_c", 0.0) > 70.0
                    or metrics.get("buffer_drops", 0.0) > 100.0
                )
                labels[node_id] = 1 if is_anomaly else 0
            node_feature_seqs.append(snapshot)
            label_seqs.append(labels)

        if not node_feature_seqs:
            return

        try:
            loss_history = self.gnn_detector.train(node_feature_seqs, label_seqs, epochs=10)
            if loss_history:
                logger.info("GNN training complete. final_loss={:.4f}", loss_history[-1])
        except Exception as exc:
            logger.warning("GNN training failed: {}", exc)

    def _validate_baseline_df(self, baseline_df: pd.DataFrame) -> None:
        missing = _REQUIRED_BASELINE_COLUMNS - set(baseline_df.columns)
        if missing:
            cols = ", ".join(sorted(missing))
            raise ValueError(f"baseline_df missing required columns: {cols}")

    def _build_latency_baselines(self) -> dict[str, float]:
        latencies: dict[str, float] = {}
        for link in self.topology.get_all_links():
            link_id = str(link["link_id"])
            latencies[link_id] = float(link.get("latency_baseline_ms", 5.0))

        for entity_id, metric_map in self.baselines.items():
            if "latency_ms" in metric_map:
                latencies[entity_id] = float(metric_map["latency_ms"])

        return latencies

    def _warm_ewma_detectors(self, baseline_df: pd.DataFrame) -> None:
        for row in baseline_df.itertuples(index=False):
            entity_id = str(getattr(row, "entity_id"))
            entity_type = str(getattr(row, "entity_type"))
            metrics = _NODE_METRICS if entity_type == "node" else _LINK_METRICS

            for metric in metrics:
                if not hasattr(row, metric):
                    continue
                value = getattr(row, metric)
                if value is None or pd.isna(value):
                    continue

                numeric = float(value)
                ewma_key = self._ewma_key(entity_id, metric)
                self.ewma_detect.update(ewma_key, numeric)
                self.ewma_ingest.update(ewma_key, numeric)

    def _init_series_forecasters(self, baseline_df: pd.DataFrame) -> None:
        self._series_forecasters.clear()
        self._series_metadata.clear()

        lstm_models: list[LSTMForecaster] = []

        link_df = baseline_df[baseline_df["entity_type"] == "link"]
        if "utilization_pct" in link_df.columns:
            for link_id, group in link_df.groupby("entity_id"):
                series = group["utilization_pct"].dropna().to_numpy(dtype=float)
                key = self._anomaly_key(str(link_id), "utilization_pct")
                model = self._train_lstm_for_series(series, str(link_id))
                self._series_forecasters[key] = model
                self._series_metadata[key] = ("link", str(link_id), "utilization_pct")
                if isinstance(model, LSTMForecaster):
                    lstm_models.append(model)

        node_df = baseline_df[baseline_df["entity_type"] == "node"]
        for node_id, group in node_df.groupby("entity_id"):
            for metric in _FORECAST_NODE_METRICS:
                if metric not in group.columns:
                    continue
                key = self._anomaly_key(str(node_id), metric)
                self._series_forecasters[key] = FallbackForecaster()
                self._series_metadata[key] = ("node", str(node_id), metric)

        self.forecaster = lstm_models[0] if lstm_models else FallbackForecaster()

    def _train_lstm_for_series(
        self, series: np.ndarray, entity_id: str
    ) -> LSTMForecaster | FallbackForecaster:
        seq_length = 30
        horizon = 10
        if len(series) < seq_length + horizon:
            logger.warning(
                "Insufficient samples for LSTM training on {} (need {}, got {}). Using fallback.",
                entity_id,
                seq_length + horizon,
                len(series),
            )
            return FallbackForecaster()

        model = LSTMForecaster(
            seq_length=seq_length,
            forecast_horizon=horizon,
            hidden_size=32,
            num_layers=1,
            lr=0.001,
            dropout=0.0,
        )
        try:
            model.train(series, epochs=8, batch_size=32)
            return model
        except Exception as exc:  # pragma: no cover - defensive for ML training failures
            logger.warning("LSTM training failed for {}. Using fallback. error={}", entity_id, exc)
            return FallbackForecaster()

    def _sanitize_telemetry(self, telemetry: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(telemetry, dict):
            raise ValueError("telemetry must be a dict")

        timestamp_raw = telemetry.get("timestamp")
        if not isinstance(timestamp_raw, str):
            raise ValueError("telemetry['timestamp'] must be an ISO datetime string")
        _ = self._parse_timestamp(timestamp_raw)

        nodes = telemetry.get("nodes")
        links = telemetry.get("links")
        if not isinstance(nodes, dict) or not isinstance(links, dict):
            raise ValueError("telemetry must contain 'nodes' and 'links' dicts")

        sanitized_nodes: dict[str, dict[str, float]] = {}
        for node_id, metrics in nodes.items():
            if not isinstance(metrics, dict):
                continue
            clean: dict[str, float] = {}
            for metric_name in _NODE_METRICS:
                value = metrics.get(metric_name)
                if isinstance(value, (int, float)):
                    clean[metric_name] = float(value)
            if clean:
                sanitized_nodes[str(node_id)] = clean

        sanitized_links: dict[str, dict[str, float]] = {}
        for link_id, metrics in links.items():
            if not isinstance(metrics, dict):
                continue
            clean = {}
            for metric_name in _LINK_METRICS:
                value = metrics.get(metric_name)
                if isinstance(value, (int, float)):
                    clean[metric_name] = float(value)
            if clean:
                sanitized_links[str(link_id)] = clean

        return {
            "timestamp": timestamp_raw,
            "nodes": sanitized_nodes,
            "links": sanitized_links,
        }

    def _detect_predicted_anomalies(self) -> list[Anomaly]:
        if not self.history:
            return []

        timestamp = self._parse_timestamp(self.history[-1]["timestamp"])
        predicted: list[Anomaly] = []

        for series_key, model in self._series_forecasters.items():
            entity_type, entity_id, metric_name = self._series_metadata[series_key]
            threshold = _FORECAST_THRESHOLDS.get(metric_name)
            if threshold is None:
                continue

            values = self._extract_series(entity_type, entity_id, metric_name)
            if not values:
                continue

            recent = np.array(values, dtype=float)
            try:
                if isinstance(model, LSTMForecaster):
                    if len(recent) < model.seq_length:
                        continue
                    # Collect MC dropout uncertainty score across all LSTM series.
                    try:
                        unc_result = model.predict_with_uncertainty(recent, n_samples=20)
                        unc_score = float(unc_result.get("uncertainty_score", 0.0))
                        if unc_score > self._last_uncertainty_score:
                            self._last_uncertainty_score = unc_score
                    except Exception:
                        pass
                    forecast = model.predict_congestion(recent, threshold=threshold)
                    is_fallback = False
                else:
                    forecast = model.predict_congestion(recent, threshold=threshold, horizon=10)
                    is_fallback = True
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                logger.warning(
                    "Forecast failed for {} ({}). error={}",
                    series_key,
                    type(model).__name__,
                    exc,
                )
                continue

            if not bool(forecast.get("will_congest", False)):
                continue

            peak = float(forecast.get("predicted_peak", recent[-1]))
            minutes_until_raw = forecast.get("minutes_until")
            minutes_until = (
                int(minutes_until_raw)
                if isinstance(minutes_until_raw, (int, float)) and minutes_until_raw is not None
                else 10
            )
            predictions = forecast.get("predictions")
            prediction_values = (
                [float(v) for v in predictions]
                if isinstance(predictions, list) and predictions
                else [peak]
            )

            confidence = self._forecast_confidence(
                prediction_values=prediction_values,
                threshold=threshold,
                predicted_peak=peak,
                minutes_until=minutes_until,
                is_fallback=is_fallback,
            )
            severity = self._severity_from_ratio(peak / threshold if threshold > 0 else 1.0)

            predicted.append(
                Anomaly(
                    timestamp=timestamp,
                    metric_name=metric_name,
                    node_id=entity_id if entity_type == "node" else None,
                    link_id=entity_id if entity_type == "link" else None,
                    observed_value=round(peak, 6),
                    expected_value=threshold,
                    severity=severity,
                    confidence=confidence,
                    detector_type="forecast_predicted",
                )
            )

        return predicted

    def _extract_series(self, entity_type: Literal["node", "link"], entity_id: str, metric: str) -> list[float]:
        values: list[float] = []
        for snapshot in self.history:
            pool = snapshot["nodes"] if entity_type == "node" else snapshot["links"]
            metrics = pool.get(entity_id)
            if not isinstance(metrics, dict):
                continue
            value = metrics.get(metric)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def _merge_cycle_anomalies(
        self, observed: list[Anomaly], predicted: list[Anomaly]
    ) -> dict[str, Anomaly]:
        merged: dict[str, Anomaly] = {}

        for anomaly in predicted:
            key = self._anomaly_key_for_anomaly(anomaly)
            merged[key] = anomaly

        for anomaly in observed:
            key = self._anomaly_key_for_anomaly(anomaly)
            existing = merged.get(key)
            if existing is None:
                merged[key] = anomaly
                continue

            if existing.detector_type == "forecast_predicted":
                # Observed signals override predictive signals for the same entity+metric.
                merged[key] = anomaly
                continue

            merged[key] = self._stronger_anomaly(existing, anomaly)

        return merged

    def _apply_cycle_to_active_state(self, cycle_anomalies: dict[str, Anomaly]) -> list[Anomaly]:
        emitted: list[Anomaly] = []
        current_keys = set(cycle_anomalies)

        for key in list(self.active_anomalies.keys()):
            if key not in current_keys:
                del self.active_anomalies[key]

        for key, anomaly in cycle_anomalies.items():
            previous = self.active_anomalies.get(key)
            if previous is None:
                self.active_anomalies[key] = anomaly
                emitted.append(anomaly)
                continue

            should_emit = (
                previous.severity != anomaly.severity
                or abs(previous.confidence - anomaly.confidence) >= 0.05
            )
            self.active_anomalies[key] = anomaly
            if should_emit:
                emitted.append(anomaly)

        emitted.sort(
            key=lambda a: (_SEVERITY_ORDER[a.severity], a.confidence),
            reverse=True,
        )
        self._anomaly_history.extend(anomaly.model_copy(deep=True) for anomaly in emitted)
        return emitted

    def _stronger_anomaly(self, left: Anomaly, right: Anomaly) -> Anomaly:
        if _SEVERITY_ORDER[right.severity] > _SEVERITY_ORDER[left.severity]:
            return right
        if _SEVERITY_ORDER[right.severity] < _SEVERITY_ORDER[left.severity]:
            return left
        return right if right.confidence >= left.confidence else left

    def _forecast_confidence(
        self,
        prediction_values: list[float],
        threshold: float,
        predicted_peak: float,
        minutes_until: int,
        is_fallback: bool,
    ) -> float:
        margin_ratio = max(0.0, (predicted_peak - threshold) / max(threshold, 1.0))
        std = float(np.std(np.array(prediction_values, dtype=float)))
        stability = max(0.0, 1.0 - min(std / 15.0, 1.0))

        if minutes_until <= 3:
            urgency = 1.0
        elif minutes_until <= 6:
            urgency = 0.75
        else:
            urgency = 0.5

        confidence = 0.45 + min(margin_ratio, 0.5) * 0.5 + stability * 0.25 + urgency * 0.2
        if is_fallback:
            confidence -= 0.15
        return round(float(np.clip(confidence, 0.0, 0.95)), 4)

    def _severity_from_ratio(self, ratio: float) -> Severity:
        if ratio >= 1.25:
            return "critical"
        if ratio >= 1.1:
            return "high"
        if ratio >= 1.0:
            return "medium"
        return "low"

    def _anomaly_key_for_anomaly(self, anomaly: Anomaly) -> str:
        entity = anomaly.node_id or anomaly.link_id
        if entity is None:
            raise ValueError("anomaly must contain node_id or link_id")
        return self._anomaly_key(entity, anomaly.metric_name)

    def _anomaly_key(self, entity: str, metric: str) -> str:
        return f"{entity}_{metric}"

    def _ewma_key(self, entity: str, metric: str) -> str:
        return f"{entity}:{metric}"

    def _parse_timestamp(self, timestamp: str) -> datetime:
        ts = timestamp.strip()
        if ts.endswith("Z"):
            ts = f"{ts[:-1]}+00:00"
        parsed = datetime.fromisoformat(ts)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed


def _build_baselines_from_dataframe(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute per-entity metric baselines from baseline telemetry dataframe."""
    baselines: dict[str, dict[str, float]] = {}
    metrics = [m for m in _DYNAMIC_METRICS if m in df.columns]
    grouped = df.groupby("entity_id")
    for entity_id, group in grouped:
        values: dict[str, float] = {}
        for metric in metrics:
            series = pd.to_numeric(group[metric], errors="coerce").dropna()
            if not series.empty:
                values[metric] = float(series.mean())
        baselines[str(entity_id)] = values
    return baselines


if __name__ == "__main__":
    from src.simulator.telemetry import TelemetryGenerator

    topology = NetworkTopology()
    telemetry_gen = TelemetryGenerator(topology, seed=42)
    baseline_df = telemetry_gen.generate_baseline(duration_hours=2)
    baselines = _build_baselines_from_dataframe(baseline_df)

    observer = ObserverAgent(topology=topology, baselines=baselines)
    observer.train_detectors(baseline_df)

    print("Feeding observer with dummy telemetry and printing emitted anomalies:")
    for idx in range(20):
        snapshot = telemetry_gen.step()

        # Add synthetic spikes in the final steps to surface anomalies.
        if idx >= 14 and "CR1-CR2" in snapshot["links"]:
            snapshot["links"]["CR1-CR2"]["utilization_pct"] = 96.0
            snapshot["links"]["CR1-CR2"]["latency_ms"] = 24.0
            snapshot["links"]["CR1-CR2"]["packet_loss_pct"] = 2.5
            snapshot["links"]["CR1-CR2"]["throughput_gbps"] = 192.0
        if idx >= 16 and "CR1" in snapshot["nodes"]:
            snapshot["nodes"]["CR1"]["cpu_pct"] = 95.0
            snapshot["nodes"]["CR1"]["memory_pct"] = 91.0
            snapshot["nodes"]["CR1"]["temperature_c"] = 75.0

        observer.ingest(snapshot)
        anomalies = observer.detect()
        if anomalies:
            print(f"\n[t+{idx + 1:02d}] emitted {len(anomalies)} anomalies")
            for anomaly in anomalies:
                entity = anomaly.node_id or anomaly.link_id
                print(
                    f"  - [{anomaly.severity:8s}] {anomaly.metric_name:16s} "
                    f"entity={entity:10s} observed={anomaly.observed_value:>7.2f} "
                    f"expected={anomaly.expected_value:>7.2f} conf={anomaly.confidence:.2f} "
                    f"type={anomaly.detector_type}"
                )

    print("\nActive anomaly count:", len(observer.get_active_anomalies()))
    print("Health summary:", observer.get_network_health_summary())
