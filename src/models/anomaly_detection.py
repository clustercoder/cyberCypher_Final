from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.models.schemas import Anomaly

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEVERITY_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _severity_from_ratio(
    ratio: float,
) -> Literal["low", "medium", "high", "critical"]:
    """Map (value / threshold) ratio to severity."""
    if ratio >= 1.3:
        return "critical"
    if ratio >= 1.1:
        return "high"
    return "medium"


def _severity_from_sigmas(sigmas: float) -> Literal["low", "medium", "high", "critical"]:
    if sigmas >= 5.0:
        return "critical"
    if sigmas >= 4.0:
        return "high"
    return "medium"


def _make_anomaly(
    metric_name: str,
    observed: float,
    expected: float,
    severity: Literal["low", "medium", "high", "critical"],
    confidence: float,
    detector_type: str,
    node_id: Optional[str],
    link_id: Optional[str],
) -> Anomaly:
    return Anomaly(
        timestamp=_now(),
        metric_name=metric_name,
        node_id=node_id,
        link_id=link_id,
        observed_value=round(observed, 6),
        expected_value=round(expected, 6),
        severity=severity,
        confidence=min(0.99, max(0.0, round(confidence, 4))),
        detector_type=detector_type,
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AnomalyDetector(ABC):
    """Common interface for all anomaly detectors."""

    @abstractmethod
    def detect(
        self,
        metric_name: str,
        value: float,
        node_id: Optional[str] = None,
        link_id: Optional[str] = None,
    ) -> Optional[Anomaly]:
        """Return an Anomaly if the reading is anomalous, else None."""


# ---------------------------------------------------------------------------
# 1. ThresholdDetector
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "utilization_pct": 85.0,
    "latency_ms_factor": 2.5,   # multiplier on baseline; handled separately
    "packet_loss_pct": 1.0,
    "cpu_pct": 90.0,
    "temperature_c": 70.0,
    "buffer_drops": 100.0,
    "memory_pct": 85.0,
    "throughput_gbps": float("inf"),  # only latency factor matters for links
}


class ThresholdDetector(AnomalyDetector):
    """
    Rule-based detector.  Latency is checked as a multiple of per-link
    baseline; all other metrics are checked against absolute thresholds.
    """

    def __init__(
        self,
        thresholds: Optional[dict[str, float]] = None,
        latency_baselines: Optional[dict[str, float]] = None,
    ) -> None:
        self._thresholds: dict[str, float] = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
        # link_id → baseline latency in ms
        self._latency_baselines: dict[str, float] = latency_baselines or {}

    def detect(
        self,
        metric_name: str,
        value: float,
        node_id: Optional[str] = None,
        link_id: Optional[str] = None,
    ) -> Optional[Anomaly]:
        threshold = self._get_threshold(metric_name, link_id)
        if threshold is None or math.isinf(threshold):
            return None

        if value <= threshold:
            return None

        ratio = value / threshold
        severity = _severity_from_ratio(ratio)
        return _make_anomaly(
            metric_name=metric_name,
            observed=value,
            expected=threshold,
            severity=severity,
            confidence=0.7,
            detector_type="threshold",
            node_id=node_id,
            link_id=link_id,
        )

    def _get_threshold(self, metric_name: str, link_id: Optional[str]) -> Optional[float]:
        if metric_name == "latency_ms":
            baseline = self._latency_baselines.get(link_id or "", 5.0)
            return baseline * self._thresholds.get("latency_ms_factor", 2.5)
        return self._thresholds.get(metric_name)


# ---------------------------------------------------------------------------
# 2. EWMADetector
# ---------------------------------------------------------------------------

class _EWMAState:
    """Per-metric online EWMA + variance tracker."""

    def __init__(self, span: int) -> None:
        self._alpha = 2.0 / (span + 1)
        self._mu: Optional[float] = None
        self._sigma2: float = 0.0
        self._n: int = 0
        self._warmup = span // 2

    def update(self, x: float) -> None:
        self._n += 1
        if self._mu is None:
            self._mu = x
            return
        diff = x - self._mu
        self._mu += self._alpha * diff
        self._sigma2 = (1 - self._alpha) * (self._sigma2 + self._alpha * diff * diff)

    @property
    def warmed_up(self) -> bool:
        return self._n >= self._warmup

    @property
    def mu(self) -> float:
        return self._mu if self._mu is not None else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self._sigma2))


class EWMADetector(AnomalyDetector):
    """
    Exponentially Weighted Moving Average detector.
    Flags values outside ewma ± 3·std.  Confidence scales with sigma deviation.
    """

    def __init__(self, span: int = 20) -> None:
        self._span = span
        self._states: dict[str, _EWMAState] = {}

    def _key(
        self, metric_name: str, node_id: Optional[str], link_id: Optional[str]
    ) -> str:
        entity = node_id or link_id or "global"
        return f"{entity}:{metric_name}"

    def update(self, metric_key: str, value: float) -> None:
        if metric_key not in self._states:
            self._states[metric_key] = _EWMAState(self._span)
        self._states[metric_key].update(value)

    def detect(
        self,
        metric_name: str,
        value: float,
        node_id: Optional[str] = None,
        link_id: Optional[str] = None,
    ) -> Optional[Anomaly]:
        key = self._key(metric_name, node_id, link_id)
        if key not in self._states:
            self._states[key] = _EWMAState(self._span)

        state = self._states[key]
        # Check before updating so current value doesn't pollute the baseline
        result = self._check(metric_name, value, node_id, link_id, state)
        state.update(value)
        return result

    def _check(
        self,
        metric_name: str,
        value: float,
        node_id: Optional[str],
        link_id: Optional[str],
        state: _EWMAState,
    ) -> Optional[Anomaly]:
        if not state.warmed_up:
            return None

        std = state.std
        if std < 1e-6:
            return None

        mu = state.mu
        sigmas = abs(value - mu) / std

        if sigmas < 3.0:
            return None

        confidence = min(0.95, sigmas / 3.0 * 0.7)
        severity = _severity_from_sigmas(sigmas)
        return _make_anomaly(
            metric_name=metric_name,
            observed=value,
            expected=mu,
            severity=severity,
            confidence=confidence,
            detector_type="ewma",
            node_id=node_id,
            link_id=link_id,
        )


# ---------------------------------------------------------------------------
# 3. IsolationForestDetector
# ---------------------------------------------------------------------------

class IsolationForestDetector(AnomalyDetector):
    """
    One IsolationForest model per metric column, trained on baseline data.
    After training, detects anomalies in single-value samples.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self._contamination = contamination
        self._random_state = random_state
        # metric_name → fitted IsolationForest
        self._models: dict[str, IsolationForest] = {}
        self._trained = False

    def train(self, data: pd.DataFrame) -> None:
        """Fit one IF per numeric column in ``data``."""
        numeric_cols = data.select_dtypes(include="number").columns
        for col in numeric_cols:
            series = data[col].dropna().values.reshape(-1, 1)
            if len(series) < 10:
                continue
            model = IsolationForest(
                contamination=self._contamination,
                random_state=self._random_state,
                n_estimators=100,
            )
            model.fit(series)
            self._models[col] = model
        self._trained = True

    def detect(
        self,
        metric_name: str,
        value: float,
        node_id: Optional[str] = None,
        link_id: Optional[str] = None,
    ) -> Optional[Anomaly]:
        if not self._trained:
            return None
        model = self._models.get(metric_name)
        if model is None:
            return None

        sample = np.array([[value]])
        prediction = model.predict(sample)[0]
        if prediction != -1:
            return None

        # decision_function: more negative → more anomalous (typical range ~-0.5 to +0.5)
        score = float(model.decision_function(sample)[0])
        # Map score to confidence: score=-0.5 → ~0.95, score=-0.1 → ~0.55
        confidence = min(0.95, max(0.5, 0.5 - score))
        severity = self._severity_from_score(score)
        # Expected value: use the mean of training data (from offset_)
        expected = float(-model.offset_)

        return _make_anomaly(
            metric_name=metric_name,
            observed=value,
            expected=expected,
            severity=severity,
            confidence=confidence,
            detector_type="isolation_forest",
            node_id=node_id,
            link_id=link_id,
        )

    @staticmethod
    def _severity_from_score(score: float) -> Literal["low", "medium", "high", "critical"]:
        if score <= -0.3:
            return "critical"
        if score <= -0.2:
            return "high"
        if score <= -0.1:
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class AnomalyDetectorEnsemble:
    """
    Runs multiple detectors in parallel, deduplicates results, and boosts
    confidence when 2+ detectors agree on the same (metric, entity).
    """

    def __init__(self, detectors: list[AnomalyDetector]) -> None:
        self._detectors = detectors

    def detect_all(
        self, readings: dict[str, Any]
    ) -> list[Anomaly]:
        """
        ``readings`` format:
        {
          "nodes": {node_id: {"cpu_pct": ..., "memory_pct": ..., ...}},
          "links": {link_id: {"utilization_pct": ..., "latency_ms": ..., ...}},
        }
        """
        raw: list[Anomaly] = []

        for node_id, metrics in readings.get("nodes", {}).items():
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                for detector in self._detectors:
                    result = detector.detect(metric_name, float(value), node_id=node_id)
                    if result is not None:
                        raw.append(result)

        for link_id, metrics in readings.get("links", {}).items():
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                for detector in self._detectors:
                    result = detector.detect(metric_name, float(value), link_id=link_id)
                    if result is not None:
                        raw.append(result)

        return self._deduplicate_and_boost(raw)

    def _deduplicate_and_boost(self, anomalies: list[Anomaly]) -> list[Anomaly]:
        # Group by (metric_name, node_id, link_id)
        groups: dict[tuple[str, Optional[str], Optional[str]], list[Anomaly]] = {}
        for a in anomalies:
            key = (a.metric_name, a.node_id, a.link_id)
            groups.setdefault(key, []).append(a)

        result: list[Anomaly] = []
        for detections in groups.values():
            # Pick highest-confidence detection as canonical
            best = max(detections, key=lambda a: a.confidence)
            if len(detections) >= 2:
                # Multi-detector agreement: boost confidence
                boosted = min(0.99, best.confidence + 0.15)
                best = best.model_copy(update={"confidence": round(boosted, 4)})
            result.append(best)

        # Sort: severity desc, then confidence desc
        result.sort(
            key=lambda a: (_SEVERITY_ORDER[a.severity], a.confidence),
            reverse=True,
        )
        return result


# ---------------------------------------------------------------------------
# Main — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.simulator.telemetry import TelemetryGenerator
    from src.simulator.topology import NetworkTopology

    topo = NetworkTopology()
    gen = TelemetryGenerator(topo, seed=42)

    print("Generating 6 hours of baseline telemetry for training…")
    df = gen.generate_baseline(duration_hours=6)
    print(f"  Rows: {len(df):,}\n")

    # ── ThresholdDetector ─────────────────────────────────────────────────
    latency_baselines = {
        link["link_id"]: link["latency_baseline_ms"]
        for link in topo.get_all_links()
    }
    td = ThresholdDetector(latency_baselines=latency_baselines)

    samples_threshold = [
        ("utilization_pct", 92.0, None, "CR1-CR2"),
        ("packet_loss_pct", 0.5,  None, "CR1-CR2"),   # below threshold
        ("latency_ms",      18.0, None, "CR1-CR2"),   # 5ms baseline × 2.5 = 12.5ms
        ("cpu_pct",         95.0, "CR1", None),
        ("temperature_c",   65.0, "AGG1", None),      # below 70
    ]

    print("=== ThresholdDetector ===")
    for metric, val, nid, lid in samples_threshold:
        a = td.detect(metric, val, node_id=nid, link_id=lid)
        entity = nid or lid
        if a:
            print(f"  ANOMALY  [{a.severity:8s}] {metric:20s} = {val:>8.2f}  "
                  f"entity={entity}  conf={a.confidence:.2f}")
        else:
            print(f"  normal   [        ] {metric:20s} = {val:>8.2f}  entity={entity}")

    # ── EWMADetector ──────────────────────────────────────────────────────
    ewma = EWMADetector(span=20)

    # Warm up on normal link data
    link_df = df[(df["entity_id"] == "CR1-CR2") & (df["entity_type"] == "link")]
    print("\n=== EWMADetector (warming up on CR1-CR2 utilization) ===")
    for val in link_df["utilization_pct"].head(30):
        ewma.detect("utilization_pct", float(val), link_id="CR1-CR2")

    # Inject a spike
    spike_tests = [55.0, 57.0, 99.0, 100.0, 58.0]
    for val in spike_tests:
        a = ewma.detect("utilization_pct", val, link_id="CR1-CR2")
        if a:
            print(f"  ANOMALY  [{a.severity:8s}] utilization_pct = {val:>6.1f}%  "
                  f"expected≈{a.expected_value:.1f}  conf={a.confidence:.2f}")
        else:
            print(f"  normal             utilization_pct = {val:>6.1f}%")

    # ── IsolationForestDetector ───────────────────────────────────────────
    print("\n=== IsolationForestDetector (trained on 6h baseline) ===")
    link_train = df[df["entity_type"] == "link"][
        ["utilization_pct", "latency_ms", "packet_loss_pct", "throughput_gbps"]
    ]
    ifd = IsolationForestDetector(contamination=0.05)
    ifd.train(link_train)
    print("  Trained on link metrics.")

    if_tests = [
        ("utilization_pct", 55.0,  "CR1-CR2"),  # normal
        ("utilization_pct", 98.0,  "CR1-CR2"),  # high
        ("latency_ms",       5.5,  "CR1-CR2"),  # normal
        ("latency_ms",      45.0,  "CR1-CR2"),  # very high
        ("packet_loss_pct",  0.01, "CR1-CR2"),  # normal
        ("packet_loss_pct",  8.0,  "CR1-CR2"),  # severe
    ]
    for metric, val, lid in if_tests:
        a = ifd.detect(metric, val, link_id=lid)
        if a:
            print(f"  ANOMALY  [{a.severity:8s}] {metric:20s} = {val:>7.3f}  "
                  f"conf={a.confidence:.2f}")
        else:
            print(f"  normal   [        ] {metric:20s} = {val:>7.3f}")

    # ── Ensemble ─────────────────────────────────────────────────────────
    print("\n=== AnomalyDetectorEnsemble ===")
    ensemble = AnomalyDetectorEnsemble([td, ewma, ifd])

    readings = {
        "nodes": {
            "CR1": {"cpu_pct": 93.0, "memory_pct": 78.0, "temperature_c": 55.0, "buffer_drops": 5},
        },
        "links": {
            "CR1-CR2": {
                "utilization_pct": 96.0,
                "latency_ms": 22.0,
                "packet_loss_pct": 3.5,
                "throughput_gbps": 192.0,
                "capacity_gbps": 200.0,
            },
        },
    }

    anomalies = ensemble.detect_all(readings)
    print(f"  Detected {len(anomalies)} anomalies (sorted by severity then confidence):\n")
    for a in anomalies:
        entity = a.node_id or a.link_id
        print(f"  [{a.severity:8s}] {a.metric_name:22s} entity={entity:10s}  "
              f"obs={a.observed_value:>8.3f}  exp={a.expected_value:>8.3f}  "
              f"conf={a.confidence:.2f}  detector={a.detector_type}")

    print(f"\n✅ All detectors operational — {len(anomalies)} anomalies surfaced from ensemble")
    sys.exit(0)
