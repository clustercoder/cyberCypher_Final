from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.agents.observer import ObserverAgent
from src.models.forecasting import FallbackForecaster, LSTMForecaster
from src.models.schemas import Anomaly
from src.simulator.topology import NetworkTopology


def _baseline_df(periods: int = 70) -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, float | str]] = []
    for i in range(periods):
        ts = (start + timedelta(minutes=i)).isoformat()
        rows.append(
            {
                "timestamp": ts,
                "entity_id": "CR1",
                "entity_type": "node",
                "cpu_pct": 45.0 + i * 0.2,
                "memory_pct": 50.0 + i * 0.15,
                "temperature_c": 40.0 + i * 0.05,
                "buffer_drops": float(2 + i % 3),
            }
        )
        rows.append(
            {
                "timestamp": ts,
                "entity_id": "CR1-CR2",
                "entity_type": "link",
                "utilization_pct": 55.0 + i * 0.25,
                "latency_ms": 5.0 + i * 0.01,
                "packet_loss_pct": 0.02 + i * 0.001,
                "throughput_gbps": 100.0 + i * 0.2,
                "capacity_gbps": 200.0,
                "latency_baseline_ms": 5.0,
            }
        )
    return pd.DataFrame(rows)


def _baselines_from_df(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    metrics = [
        "cpu_pct",
        "memory_pct",
        "temperature_c",
        "buffer_drops",
        "utilization_pct",
        "latency_ms",
        "packet_loss_pct",
        "throughput_gbps",
    ]
    for entity_id, group in df.groupby("entity_id"):
        per_metric: dict[str, float] = {}
        for metric in metrics:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                if not values.empty:
                    per_metric[metric] = float(values.mean())
        result[str(entity_id)] = per_metric
    return result


def _snapshot(ts: datetime, util: float = 55.0, cpu: float = 50.0) -> dict[str, object]:
    return {
        "timestamp": ts.isoformat(),
        "nodes": {
            "CR1": {
                "cpu_pct": cpu,
                "memory_pct": min(100.0, cpu * 0.9),
                "temperature_c": 45.0 + max(0.0, cpu - 50.0) * 0.4,
                "buffer_drops": 5.0,
            }
        },
        "links": {
            "CR1-CR2": {
                "utilization_pct": util,
                "latency_ms": 5.0 + max(0.0, util - 55.0) * 0.08,
                "packet_loss_pct": max(0.0, (util - 80.0) * 0.03),
                "throughput_gbps": min(200.0, util * 2.0),
            }
        },
    }


def _make_anomaly(
    metric: str,
    severity: str,
    confidence: float,
    *,
    node_id: str | None = None,
    link_id: str | None = None,
    detector_type: str = "threshold",
) -> Anomaly:
    return Anomaly(
        timestamp=datetime.now(timezone.utc),
        metric_name=metric,
        node_id=node_id,
        link_id=link_id,
        observed_value=95.0,
        expected_value=85.0,
        severity=severity,  # type: ignore[arg-type]
        confidence=confidence,
        detector_type=detector_type,
    )


def test_ingest_history_keeps_last_60_snapshots() -> None:
    topo = NetworkTopology()
    observer = ObserverAgent(topology=topo, baselines={})
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for i in range(70):
        observer.ingest(_snapshot(start + timedelta(minutes=i), util=50.0 + i * 0.2, cpu=48.0 + i * 0.1))

    assert len(observer.history) == 60
    assert observer.history[0]["timestamp"] == (start + timedelta(minutes=10)).isoformat()


def test_train_detectors_initializes_models_and_states(monkeypatch) -> None:
    topo = NetworkTopology()
    df = _baseline_df(periods=70)
    baselines = _baselines_from_df(df)
    observer = ObserverAgent(topology=topo, baselines=baselines)

    def fake_train(self: LSTMForecaster, historical_data: np.ndarray, epochs: int = 50, batch_size: int = 32) -> list[float]:
        self._scaler.fit(historical_data.reshape(-1, 1))
        self._trained = True
        return [0.01]

    monkeypatch.setattr(LSTMForecaster, "train", fake_train)
    observer.train_detectors(df)

    assert observer.isolation_forest_detector._trained
    assert observer._series_forecasters
    assert any(isinstance(model, LSTMForecaster) for model in observer._series_forecasters.values())
    assert isinstance(observer.forecaster, LSTMForecaster)
    assert observer.ewma_detect._states
    assert observer.ewma_ingest._states


def test_detect_dedup_and_update_behavior() -> None:
    topo = NetworkTopology()
    observer = ObserverAgent(topology=topo, baselines={})
    observer.ingest(_snapshot(datetime(2026, 1, 1, tzinfo=timezone.utc), util=95.0, cpu=55.0))
    observer._series_forecasters.clear()
    observer._series_metadata.clear()

    first = _make_anomaly("utilization_pct", "high", 0.70, link_id="CR1-CR2")
    second_small_delta = _make_anomaly("utilization_pct", "high", 0.73, link_id="CR1-CR2")
    third_large_delta = _make_anomaly("utilization_pct", "high", 0.82, link_id="CR1-CR2")

    observer.ensemble.detect_all = lambda _r: [first]  # type: ignore[method-assign]
    emitted_first = observer.detect()
    assert len(emitted_first) == 1

    observer.ensemble.detect_all = lambda _r: [second_small_delta]  # type: ignore[method-assign]
    emitted_second = observer.detect()
    assert emitted_second == []

    observer.ensemble.detect_all = lambda _r: [third_large_delta]  # type: ignore[method-assign]
    emitted_third = observer.detect()
    assert len(emitted_third) == 1
    assert abs(emitted_third[0].confidence - 0.82) < 1e-6


def test_detect_resolves_anomalies_when_cleared() -> None:
    topo = NetworkTopology()
    observer = ObserverAgent(topology=topo, baselines={})
    observer.ingest(_snapshot(datetime(2026, 1, 1, tzinfo=timezone.utc), util=95.0, cpu=55.0))
    observer._series_forecasters.clear()
    observer._series_metadata.clear()

    anomaly = _make_anomaly("utilization_pct", "high", 0.85, link_id="CR1-CR2")
    observer.ensemble.detect_all = lambda _r: [anomaly]  # type: ignore[method-assign]
    observer.detect()
    assert observer.get_active_anomalies()

    observer.ensemble.detect_all = lambda _r: []  # type: ignore[method-assign]
    emitted = observer.detect()
    assert emitted == []
    assert observer.get_active_anomalies() == []


def test_forecasting_emits_predicted_anomalies_for_link_and_node_metrics() -> None:
    topo = NetworkTopology()
    observer = ObserverAgent(topology=topo, baselines={})
    observer.ensemble.detect_all = lambda _r: []  # type: ignore[method-assign]

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(35):
        snap = _snapshot(
            start + timedelta(minutes=i),
            util=74.0 + i * 0.55,
            cpu=78.0 + i * 0.45,
        )
        snap["nodes"]["CR1"]["temperature_c"] = 64.0 + i * 0.35
        observer.ingest(
            snap
        )

    observer._series_forecasters = {
        "CR1-CR2_utilization_pct": FallbackForecaster(),
        "CR1_cpu_pct": FallbackForecaster(),
        "CR1_memory_pct": FallbackForecaster(),
        "CR1_temperature_c": FallbackForecaster(),
    }
    observer._series_metadata = {
        "CR1-CR2_utilization_pct": ("link", "CR1-CR2", "utilization_pct"),
        "CR1_cpu_pct": ("node", "CR1", "cpu_pct"),
        "CR1_memory_pct": ("node", "CR1", "memory_pct"),
        "CR1_temperature_c": ("node", "CR1", "temperature_c"),
    }

    emitted = observer.detect()
    predicted = [a for a in emitted if a.detector_type == "forecast_predicted"]

    keys = {(a.node_id or a.link_id, a.metric_name) for a in predicted}
    assert ("CR1-CR2", "utilization_pct") in keys
    assert ("CR1", "cpu_pct") in keys
    assert ("CR1", "memory_pct") in keys
    assert ("CR1", "temperature_c") in keys


def test_network_health_summary_state_transitions() -> None:
    topo = NetworkTopology()
    observer = ObserverAgent(topology=topo, baselines={})

    observer.active_anomalies = {
        "CR1_cpu_pct": _make_anomaly("cpu_pct", "low", 0.62, node_id="CR1"),
    }
    healthy = observer.get_network_health_summary()
    assert healthy["overall_health"] == "healthy"
    assert healthy["worst_entity"] == "CR1"

    observer.active_anomalies = {
        "CR1_cpu_pct": _make_anomaly("cpu_pct", "medium", 0.70, node_id="CR1"),
        "CR1-CR2_utilization_pct": _make_anomaly(
            "utilization_pct", "high", 0.88, link_id="CR1-CR2"
        ),
    }
    degraded = observer.get_network_health_summary()
    assert degraded["overall_health"] == "degraded"
    assert degraded["worst_entity"] == "CR1-CR2"

    observer.active_anomalies["CR2_cpu_pct"] = _make_anomaly(
        "cpu_pct", "critical", 0.91, node_id="CR2"
    )
    critical = observer.get_network_health_summary()
    assert critical["overall_health"] == "critical"
    assert critical["worst_entity"] == "CR2"
