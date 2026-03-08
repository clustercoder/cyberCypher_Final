"""src/evaluation/generate_dataset.py

Generates a comprehensive labeled dataset for evaluating the CyberCypher agent.

Run:
    python -m src.evaluation.generate_dataset
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root on sys.path when invoked as -m
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.simulator.anomaly_injector import AnomalyInjector, ScenarioLabel
from src.simulator.telemetry import TelemetryGenerator
from src.simulator.topology import NetworkTopology

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
TRAIN_HOURS = 16
TOTAL_HOURS = 24
DATA_DIR = _ROOT / "src" / "data"

# Severity precedence for resolving overlapping scenarios at the same timestamp
_SEVERITY_RANK: dict[str, int] = {"critical": 4, "high": 3, "medium": 2, "low": 1}

# ─────────────────────────────────────────────────────────────────────────────
# Scenario plan
# ─────────────────────────────────────────────────────────────────────────────
#
# Each tuple: (hour_offset: float, scenario_type: str, kwargs: dict)
# kwargs are forwarded directly to the inject_* method.
#
# Gap verification (scenario end → next scenario start, all ≥ 30 min):
#   1. 01:00 + 45 m → 01:45  |  → 03:00 = 75 min  ✓
#   2. 03:00 + 90 m → 04:30  |  → 05:30 = 60 min  ✓
#   3. 05:30 + 30 m → 06:00  |  → 08:00 = 120 min ✓
#   4. 08:00 + 60 m → 09:00  |  → 11:00 = 120 min ✓
#   5. 11:00 + 90 m → 12:30  |  → 16:30 = 240 min ✓  ← train/test boundary at 16:00
#   6. 16:30 + 45 m → 17:15  |  → 18:00 =  45 min ✓
#   7. 18:00 + 90 m → 19:30  |  → 20:30 =  60 min ✓
#   8. 20:30 + 30 m → 21:00  |  → 21:30 =  30 min ✓
#   9. 21:30 + 60 m → 22:30  |  → 23:00 =  30 min ✓
#  10. 23:00 + 60 m → 24:00  |  end of window         ✓

_SCENARIO_PLAN: list[tuple[float, str, dict[str, Any]]] = [
    # ── Train set (hours 0 – 16) ──────────────────────────────────────────
    (1.0,  "congestion_cascade",   {"root_link": "CR1-CR2",      "duration_minutes": 45}),
    (3.0,  "hardware_degradation", {"root_node": "AGG1",         "duration_minutes": 90}),
    (5.5,  "ddos_surge",           {"target_node": "EDGE1",      "duration_minutes": 30}),
    (8.0,  "misconfiguration",     {"target_link": "AGG2-EDGE2", "duration_minutes": 60}),
    (11.0, "fiber_cut",            {"cut_link": "AGG3-EDGE3",    "duration_minutes": 90}),
    # ── Test set (hours 16 – 24) ──────────────────────────────────────────
    (16.5, "congestion_cascade",   {"root_link": "AGG1-PEER1",   "duration_minutes": 45}),
    (18.0, "hardware_degradation", {"root_node": "CR2",          "duration_minutes": 90}),
    (20.5, "ddos_surge",           {"target_node": "EDGE4",      "duration_minutes": 30}),
    (21.5, "misconfiguration",     {"target_link": "AGG3-AGG4",  "duration_minutes": 60}),
    (23.0, "fiber_cut",            {"cut_link": "AGG4-EDGE4",    "duration_minutes": 60}),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _midnight_utc() -> datetime:
    """Return today's midnight in UTC."""
    today = datetime.now(timezone.utc).date()
    return datetime(today.year, today.month, today.day, tzinfo=timezone.utc)


def _build_ground_truth(
    timestamps: pd.DatetimeIndex,
    labels: list[ScenarioLabel],
) -> pd.DataFrame:
    """
    Build a per-timestamp ground truth DataFrame.

    For each unique timestamp:
      - is_anomaly        : True if timestamp falls inside any scenario window
      - scenario_type     : type of the most severe overlapping scenario
      - root_cause        : human-readable root cause string
      - severity          : low / medium / high / critical
      - affected_entities : JSON array of affected node + link IDs
    """
    rows: list[dict[str, Any]] = []

    for ts in timestamps:
        ts_dt: datetime = ts.to_pydatetime()

        active = [
            lbl for lbl in labels
            if lbl.start_time <= ts_dt <= lbl.end_time
        ]

        if not active:
            rows.append({
                "timestamp": ts,
                "is_anomaly": False,
                "scenario_type": None,
                "root_cause": None,
                "severity": None,
                "affected_entities": None,
            })
        else:
            # Resolve ambiguity by highest severity
            primary = max(active, key=lambda l: _SEVERITY_RANK.get(l.severity, 0))
            entities = sorted(set(primary.affected_nodes + primary.affected_links))
            rows.append({
                "timestamp": ts,
                "is_anomaly": True,
                "scenario_type": primary.scenario_type,
                "root_cause": primary.root_cause,
                "severity": primary.severity,
                "affected_entities": json.dumps(entities),
            })

    return pd.DataFrame(rows)


def _scenario_to_metadata(lbl: ScenarioLabel, split: str) -> dict[str, Any]:
    return {
        "scenario_id": lbl.scenario_id,
        "scenario_type": lbl.scenario_type,
        "split": split,
        "start_time": lbl.start_time.isoformat(),
        "end_time": lbl.end_time.isoformat(),
        "duration_minutes": int((lbl.end_time - lbl.start_time).total_seconds() / 60),
        "severity": lbl.severity,
        "root_cause": lbl.root_cause,
        "root_cause_node": lbl.root_cause_node,
        "root_cause_link": lbl.root_cause_link,
        "affected_nodes": lbl.affected_nodes,
        "affected_links": lbl.affected_links,
        "description": lbl.description,
    }


def _print_summary(
    baseline_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    labels: list[ScenarioLabel],
    train_cutoff: datetime,
) -> None:
    total_ts = gt_df["timestamp"].nunique()
    anomalous_ts = gt_df["is_anomaly"].sum()

    print("\n" + "=" * 66)
    print("  CyberCypher 5.0 — Dataset Generation Summary")
    print("=" * 66)
    print(f"\n  Total timestamps     : {total_ts:,}  (1-min intervals × 24 h)")
    print(f"  Total data rows      : {anomaly_df.shape[0]:,}  (timestamps × entities)")
    print(f"  Baseline rows        : {baseline_df.shape[0]:,}")
    print(f"  Anomaly rows         : {anomaly_df.shape[0]:,}")
    print(f"\n  Train split          : {train_df.shape[0]:,} rows  ({TRAIN_HOURS}h, up to {train_cutoff.strftime('%H:%M')} UTC)")
    print(f"  Test  split          : {test_df.shape[0]:,} rows  ({TOTAL_HOURS - TRAIN_HOURS}h, after {train_cutoff.strftime('%H:%M')} UTC)")
    print(f"\n  Anomalous timestamps : {anomalous_ts:,} / {total_ts:,}  ({100 * anomalous_ts / total_ts:.1f}%)")

    print(f"\n  {'Scenario':<22} {'Split':<6} {'Start':>6}  {'Dur':>4}  {'Sev':>8}  {'Anom-TS':>8}")
    print("  " + "-" * 62)

    for lbl in labels:
        split = "train" if lbl.start_time < train_cutoff else "test"
        start_str = lbl.start_time.strftime("%H:%M")
        dur = int((lbl.end_time - lbl.start_time).total_seconds() / 60)
        n_ts = int(
            gt_df[gt_df["scenario_type"] == lbl.scenario_type]["is_anomaly"].sum()
        )
        # Each scenario type appears twice; approximate per-label count
        n_ts_label = dur  # anomaly window = duration minutes ≈ timestamps
        print(
            f"  {lbl.scenario_type:<22} {split:<6} {start_str:>6}  {dur:>3}m  "
            f"{lbl.severity:>8}  ~{n_ts_label:>5} ts"
        )

    print(f"\n  Files saved to: {DATA_DIR}/")
    print("=" * 66 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    midnight = _midnight_utc()
    train_cutoff = midnight + timedelta(hours=TRAIN_HOURS)

    # ── Step 1: Topology ──────────────────────────────────────────────────
    print("Step 1/6  Building network topology…")
    topology = NetworkTopology()
    n_nodes = topology.get_graph().number_of_nodes()
    n_links = topology.get_graph().number_of_edges() // 2
    print(f"          {n_nodes} nodes, {n_links} undirected links")

    # ── Step 2: 24h baseline telemetry ───────────────────────────────────
    print("Step 2/6  Generating 24h baseline telemetry (1-min intervals)…")
    gen = TelemetryGenerator(topology, seed=SEED)
    baseline_df = gen.generate_baseline(duration_hours=TOTAL_HOURS)
    print(f"          {len(baseline_df):,} rows  ({baseline_df['entity_id'].nunique()} entities × {TOTAL_HOURS * 60} timesteps)")

    # ── Step 3: Inject 10 anomaly scenarios ──────────────────────────────
    print("Step 3/6  Injecting 10 labeled anomaly scenarios…")
    injector = AnomalyInjector(topology, seed=SEED + 1)
    anomaly_df = baseline_df.copy()
    labels: list[ScenarioLabel] = []

    for hour_offset, stype, kwargs in _SCENARIO_PLAN:
        start = midnight + timedelta(hours=hour_offset)
        anomaly_df, lbl = injector.inject_scenario(anomaly_df, stype, start, **kwargs)
        labels.append(lbl)
        split_tag = "train" if start < train_cutoff else "test "
        dur = int((lbl.end_time - lbl.start_time).total_seconds() / 60)
        print(
            f"          [{split_tag}] {lbl.severity:>8}  "
            f"{lbl.scenario_type:<22} @ {start.strftime('%H:%M')} UTC  ({dur} min)"
        )

    # ── Step 4: Ground truth labels ───────────────────────────────────────
    print("Step 4/6  Building ground truth labels DataFrame…")
    timestamps: pd.DatetimeIndex = pd.DatetimeIndex(
        pd.to_datetime(anomaly_df["timestamp"], utc=True).unique()
    ).sort_values()
    gt_df = _build_ground_truth(timestamps, labels)
    print(f"          {len(gt_df):,} timestep rows, {gt_df['is_anomaly'].sum()} anomalous")

    # ── Step 5: Train / test split ────────────────────────────────────────
    print("Step 5/6  Splitting into train / test sets…")
    ts_col = pd.to_datetime(anomaly_df["timestamp"], utc=True)
    train_df = anomaly_df[ts_col < train_cutoff].reset_index(drop=True)
    test_df  = anomaly_df[ts_col >= train_cutoff].reset_index(drop=True)
    print(f"          train: {len(train_df):,} rows  |  test: {len(test_df):,} rows")

    # ── Step 6: Save files ────────────────────────────────────────────────
    print("Step 6/6  Saving CSV and JSON files…")

    baseline_df.to_csv(DATA_DIR / "baseline_telemetry.csv", index=False)
    anomaly_df.to_csv(DATA_DIR / "anomaly_telemetry.csv", index=False)
    train_df.to_csv(DATA_DIR / "train_telemetry.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_telemetry.csv", index=False)
    gt_df.to_csv(DATA_DIR / "ground_truth_labels.csv", index=False)

    train_labels = [l for l in labels if l.start_time < train_cutoff]
    test_labels  = [l for l in labels if l.start_time >= train_cutoff]
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "total_hours": TOTAL_HOURS,
        "train_hours": TRAIN_HOURS,
        "test_hours": TOTAL_HOURS - TRAIN_HOURS,
        "train_cutoff_utc": train_cutoff.isoformat(),
        "n_scenarios_total": len(labels),
        "n_scenarios_train": len(train_labels),
        "n_scenarios_test": len(test_labels),
        "scenarios": (
            [_scenario_to_metadata(l, "train") for l in train_labels]
            + [_scenario_to_metadata(l, "test")  for l in test_labels]
        ),
    }
    with open(DATA_DIR / "scenario_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    files = [
        "baseline_telemetry.csv",
        "anomaly_telemetry.csv",
        "train_telemetry.csv",
        "test_telemetry.csv",
        "ground_truth_labels.csv",
        "scenario_metadata.json",
    ]
    for fname in files:
        fpath = DATA_DIR / fname
        size_kb = fpath.stat().st_size / 1024
        print(f"          {fname:<32} {size_kb:>8.1f} KB")

    _print_summary(baseline_df, anomaly_df, train_df, test_df, gt_df, labels, train_cutoff)


if __name__ == "__main__":
    generate_dataset()
