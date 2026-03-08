"""src/evaluation/evaluate.py

Evaluates the CyberCypher agent's anomaly detection and decision-making performance.

Run:
    python -m src.evaluation.evaluate
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root on sys.path when invoked as -m
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agents.observer import ObserverAgent, _build_baselines_from_dataframe
from src.causal.causal_engine import CausalEngine
from src.simulator.topology import NetworkTopology

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = _ROOT / "src" / "data"
EVAL_DIR = _ROOT / "src" / "evaluation"

_NODE_METRICS = ("cpu_pct", "memory_pct", "temperature_c", "buffer_drops")
_LINK_METRICS = ("utilization_pct", "latency_ms", "packet_loss_pct", "throughput_gbps")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Load CSVs and scenario metadata."""
    train_df = pd.read_csv(DATA_DIR / "train_telemetry.csv")
    test_df = pd.read_csv(DATA_DIR / "test_telemetry.csv")
    gt_df = pd.read_csv(DATA_DIR / "ground_truth_labels.csv")

    with open(DATA_DIR / "scenario_metadata.json") as f:
        metadata = json.load(f)

    return train_df, test_df, gt_df, metadata


def _df_to_snapshots(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert flat telemetry DataFrame to list of snapshot dicts sorted by timestamp."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    snapshots: list[dict[str, Any]] = []

    for ts, group in df.groupby("timestamp", sort=True):
        nodes: dict[str, dict[str, float]] = {}
        links: dict[str, dict[str, float]] = {}

        for row in group.itertuples(index=False):
            entity_id = str(row.entity_id)
            if row.entity_type == "node":
                metrics: dict[str, float] = {}
                for m in _NODE_METRICS:
                    val = getattr(row, m, None)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        metrics[m] = float(val)
                nodes[entity_id] = metrics
            elif row.entity_type == "link":
                metrics = {}
                for m in _LINK_METRICS:
                    val = getattr(row, m, None)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        metrics[m] = float(val)
                links[entity_id] = metrics

        snapshots.append({
            "timestamp": ts.isoformat(),
            "nodes": nodes,
            "links": links,
        })

    return snapshots


def _parse_utc(iso_str: str) -> datetime:
    ts = iso_str.strip()
    if ts.endswith("Z"):
        ts = f"{ts[:-1]}+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Detection Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _run_detection(
    train_df: pd.DataFrame,
    test_snapshots: list[dict[str, Any]],
    gt_df: pd.DataFrame,
    test_scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Train ObserverAgent on train set, run detection on test snapshots."""
    print("  Building baselines from training data…")
    topology = NetworkTopology()
    baselines = _build_baselines_from_dataframe(train_df)

    observer = ObserverAgent(topology=topology, baselines=baselines)

    print("  Training detectors on train_telemetry.csv…")
    observer.train_detectors(train_df)

    # Ground truth per timestamp (test period only)
    gt_df = gt_df.copy()
    gt_df["timestamp"] = pd.to_datetime(gt_df["timestamp"], utc=True)

    # Build lookup: timestamp ISO → is_anomaly bool
    gt_lookup: dict[str, bool] = {}
    for row in gt_df.itertuples(index=False):
        key = pd.Timestamp(row.timestamp).isoformat()
        gt_lookup[key] = bool(row.is_anomaly)

    print(f"  Running detection on {len(test_snapshots)} test snapshots…")
    tp = fp = fn = tn = 0

    # For MTTD: track first detection time per scenario
    scenario_windows: list[dict[str, Any]] = []
    for s in test_scenarios:
        scenario_windows.append({
            "scenario_id": s["scenario_id"],
            "scenario_type": s["scenario_type"],
            "start": _parse_utc(s["start_time"]),
            "end": _parse_utc(s["end_time"]),
            "first_detection": None,
            "detected": False,
        })

    detected_anomalies_by_ts: dict[str, list[Any]] = {}

    for snapshot in test_snapshots:
        observer.ingest(snapshot)
        anomalies = observer.detect()

        ts_key = snapshot["timestamp"]
        has_detection = len(anomalies) > 0
        detected_anomalies_by_ts[ts_key] = anomalies

        ground_truth_anomaly = gt_lookup.get(ts_key, False)

        if ground_truth_anomaly and has_detection:
            tp += 1
        elif ground_truth_anomaly and not has_detection:
            fn += 1
        elif not ground_truth_anomaly and has_detection:
            fp += 1
        else:
            tn += 1

        if has_detection:
            ts_dt = _parse_utc(ts_key)
            for window in scenario_windows:
                if not window["detected"] and window["start"] <= ts_dt <= window["end"]:
                    window["first_detection"] = ts_dt
                    window["detected"] = True

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # MTTD per scenario
    mttd_values: list[float] = []
    scenario_mttd: list[dict[str, Any]] = []
    for window in scenario_windows:
        if window["detected"] and window["first_detection"] is not None:
            mttd_s = (window["first_detection"] - window["start"]).total_seconds()
            mttd_values.append(mttd_s)
            scenario_mttd.append({
                "scenario_id": window["scenario_id"],
                "scenario_type": window["scenario_type"],
                "detected": True,
                "mttd_seconds": round(mttd_s, 1),
            })
        else:
            scenario_mttd.append({
                "scenario_id": window["scenario_id"],
                "scenario_type": window["scenario_type"],
                "detected": False,
                "mttd_seconds": None,
            })

    mean_mttd = float(np.mean(mttd_values)) if mttd_values else None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_mttd_seconds": round(mean_mttd, 1) if mean_mttd is not None else None,
        "scenarios_detected": sum(1 for w in scenario_windows if w["detected"]),
        "scenarios_total": len(scenario_windows),
        "per_scenario_mttd": scenario_mttd,
        "_detected_anomalies_by_ts": detected_anomalies_by_ts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Root Cause Analysis Evaluation
# ─────────────────────────────────────────────────────────────────────────────

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


def _entity_in_causal_var(causal_var: str, entity: str | None) -> bool:
    """Check if entity ID appears in a causal variable string (e.g. 'CR1-CR2_utilization')."""
    if not entity or not causal_var:
        return False
    return entity.lower() in causal_var.lower()


def _run_rca(
    test_scenarios: list[dict[str, Any]],
    detected_anomalies_by_ts: dict[str, list[Any]],
) -> dict[str, Any]:
    """Run CausalEngine RCA on clusters of detected anomalies near each scenario."""
    print("  Running root cause analysis with CausalEngine…")
    topology = NetworkTopology()
    causal_engine = CausalEngine(topology=topology)

    rca_results: list[dict[str, Any]] = []
    correct = 0
    total_with_anomalies = 0

    for scenario in test_scenarios:
        start = _parse_utc(scenario["start_time"])
        end = _parse_utc(scenario["end_time"])
        gt_node = scenario.get("root_cause_node")
        gt_link = scenario.get("root_cause_link")
        gt_entity = gt_node or gt_link

        # Collect all anomalies detected in the scenario window
        window_anomalies: list[Any] = []
        for ts_key, anomalies in detected_anomalies_by_ts.items():
            ts_dt = _parse_utc(ts_key)
            if start <= ts_dt <= end:
                window_anomalies.extend(anomalies)

        if not window_anomalies:
            rca_results.append({
                "scenario_id": scenario["scenario_id"],
                "scenario_type": scenario["scenario_type"],
                "gt_root_cause": gt_entity,
                "top_hypothesis": None,
                "top_confidence": None,
                "match": False,
                "note": "no anomalies detected in window",
            })
            continue

        total_with_anomalies += 1

        try:
            hypotheses = causal_engine.find_root_cause(window_anomalies[:20])
        except Exception as exc:
            rca_results.append({
                "scenario_id": scenario["scenario_id"],
                "scenario_type": scenario["scenario_type"],
                "gt_root_cause": gt_entity,
                "top_hypothesis": None,
                "top_confidence": None,
                "match": False,
                "note": f"causal engine error: {exc}",
            })
            continue

        if not hypotheses:
            rca_results.append({
                "scenario_id": scenario["scenario_id"],
                "scenario_type": scenario["scenario_type"],
                "gt_root_cause": gt_entity,
                "top_hypothesis": None,
                "top_confidence": None,
                "match": False,
                "note": "no hypothesis returned",
            })
            continue

        top = hypotheses[0]
        causal_var = top.root_cause if hasattr(top, "root_cause") else str(top)
        match = _entity_in_causal_var(causal_var, gt_entity)

        if match:
            correct += 1

        rca_results.append({
            "scenario_id": scenario["scenario_id"],
            "scenario_type": scenario["scenario_type"],
            "gt_root_cause": gt_entity,
            "top_hypothesis": causal_var,
            "top_confidence": round(float(top.confidence), 4) if hasattr(top, "confidence") else None,
            "match": match,
            "note": None,
        })

    rca_accuracy = correct / total_with_anomalies if total_with_anomalies > 0 else None

    return {
        "rca_accuracy": round(rca_accuracy, 4) if rca_accuracy is not None else None,
        "correct_rca": correct,
        "scenarios_with_detections": total_with_anomalies,
        "per_scenario_rca": rca_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — Decision Evaluation (mock)
# ─────────────────────────────────────────────────────────────────────────────

# Simulated MTTM constants (seconds) — representative of each autonomous action path
_SIMULATED_MTTM_BY_SCENARIO: dict[str, float] = {
    "congestion_cascade": 45.0,
    "hardware_degradation": 120.0,
    "ddos_surge": 30.0,
    "misconfiguration": 60.0,
    "fiber_cut": 90.0,
}

_EXPECTED_DECISION: dict[str, str] = {
    "congestion_cascade": "reroute_traffic",
    "hardware_degradation": "failover_node",
    "ddos_surge": "rate_limit_flow",
    "misconfiguration": "rollback_config",
    "fiber_cut": "activate_backup_path",
}

_AUTONOMY_LEVEL: dict[str, str] = {
    "congestion_cascade": "AUTOMATIC_CANARY",
    "hardware_degradation": "HUMAN_APPROVAL",
    "ddos_surge": "AUTOMATIC",
    "misconfiguration": "AUTOMATIC_CANARY",
    "fiber_cut": "HUMAN_APPROVAL",
}


def _run_decision_eval(
    test_scenarios: list[dict[str, Any]],
    detection_result: dict[str, Any],
) -> dict[str, Any]:
    """Mock decision evaluation based on detection results and scenario metadata."""
    print("  Running mock decision evaluation…")

    scenario_mttd_map = {
        s["scenario_id"]: s["mttd_seconds"]
        for s in detection_result["per_scenario_mttd"]
    }
    scenario_detected_map = {
        s["scenario_id"]: s["detected"]
        for s in detection_result["per_scenario_mttd"]
    }

    decision_results: list[dict[str, Any]] = []
    mttm_values: list[float] = []

    for scenario in test_scenarios:
        sid = scenario["scenario_id"]
        stype = scenario["scenario_type"]
        detected = scenario_detected_map.get(sid, False)
        mttd = scenario_mttd_map.get(sid)

        if not detected or mttd is None:
            decision_results.append({
                "scenario_id": sid,
                "scenario_type": stype,
                "detected": False,
                "decision": None,
                "autonomy_level": None,
                "mttm_seconds": None,
                "z3_verified": None,
                "note": "skipped — not detected",
            })
            continue

        sim_mttm = _SIMULATED_MTTM_BY_SCENARIO.get(stype, 60.0)
        mttm_values.append(sim_mttm)

        # High-risk scenarios require human approval — Z3 blocks auto-execution
        auto_levels = {"AUTOMATIC", "AUTOMATIC_CANARY"}
        autonomy = _AUTONOMY_LEVEL.get(stype, "AUTOMATIC_CANARY")
        z3_verified = autonomy in auto_levels

        decision_results.append({
            "scenario_id": sid,
            "scenario_type": stype,
            "detected": True,
            "decision": _EXPECTED_DECISION.get(stype, "unknown"),
            "autonomy_level": autonomy,
            "mttm_seconds": sim_mttm,
            "z3_verified": z3_verified,
            "note": "simulated — no live action log available",
        })

    mean_mttm = float(np.mean(mttm_values)) if mttm_values else None

    return {
        "mean_mttm_seconds": round(mean_mttm, 1) if mean_mttm is not None else None,
        "z3_pass_rate": (
            round(
                sum(1 for r in decision_results if r["z3_verified"]) /
                sum(1 for r in decision_results if r["detected"]),
                4,
            )
            if any(r["detected"] for r in decision_results)
            else None
        ),
        "per_scenario_decisions": decision_results,
        "note": "MTTM values are simulated estimates; real values require live action logs",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part 4 — Report Generation
# ─────────────────────────────────────────────────────────────────────────────


def _build_report(
    detection: dict[str, Any],
    rca: dict[str, Any],
    decision: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "seed": metadata.get("seed"),
            "total_hours": metadata.get("total_hours"),
            "train_hours": metadata.get("train_hours"),
            "test_hours": metadata.get("test_hours"),
            "n_scenarios_test": metadata.get("n_scenarios_test"),
        },
        "detection": {
            "tp": detection["tp"],
            "fp": detection["fp"],
            "fn": detection["fn"],
            "tn": detection["tn"],
            "precision": detection["precision"],
            "recall": detection["recall"],
            "f1": detection["f1"],
            "mean_mttd_seconds": detection["mean_mttd_seconds"],
            "scenarios_detected": detection["scenarios_detected"],
            "scenarios_total": detection["scenarios_total"],
            "per_scenario_mttd": detection["per_scenario_mttd"],
        },
        "root_cause_analysis": {
            "rca_accuracy": rca["rca_accuracy"],
            "correct_rca": rca["correct_rca"],
            "scenarios_with_detections": rca["scenarios_with_detections"],
            "per_scenario_rca": rca["per_scenario_rca"],
        },
        "decision": {
            "mean_mttm_seconds": decision["mean_mttm_seconds"],
            "z3_pass_rate": decision["z3_pass_rate"],
            "note": decision["note"],
            "per_scenario_decisions": decision["per_scenario_decisions"],
        },
    }


def _format_report_text(report: dict[str, Any]) -> str:
    det = report["detection"]
    rca = report["root_cause_analysis"]
    dec = report["decision"]
    ds = report["dataset"]

    def pct(v: float | None) -> str:
        return f"{v * 100:.1f}%" if v is not None else "—"

    def sec(v: float | None) -> str:
        return f"{v:.1f}s" if v is not None else "—"

    lines: list[str] = [
        "",
        "=" * 70,
        "  CyberCypher 5.0 — Evaluation Report",
        f"  Generated: {report['generated_at']}",
        "=" * 70,
        "",
        "  Dataset",
        "  -------",
        f"  Seed         : {ds['seed']}",
        f"  Train period : {ds['train_hours']}h",
        f"  Test  period : {ds['test_hours']}h",
        f"  Test scenarios: {ds['n_scenarios_test']}",
        "",
        "  Part 1 — Anomaly Detection",
        "  ---------------------------",
        f"  TP={det['tp']}  FP={det['fp']}  FN={det['fn']}  TN={det['tn']}",
        f"  Precision : {pct(det['precision'])}",
        f"  Recall    : {pct(det['recall'])}",
        f"  F1 Score  : {pct(det['f1'])}",
        f"  Mean MTTD : {sec(det['mean_mttd_seconds'])}",
        f"  Scenarios detected: {det['scenarios_detected']} / {det['scenarios_total']}",
        "",
        f"  {'Scenario ID':<14} {'Type':<22} {'Detected':>8}  {'MTTD':>8}",
        "  " + "-" * 58,
    ]
    for s in det["per_scenario_mttd"]:
        detected_str = "yes" if s["detected"] else "no"
        mttd_str = sec(s["mttd_seconds"]) if s["mttd_seconds"] is not None else "—"
        lines.append(
            f"  {s['scenario_id']:<14} {s['scenario_type']:<22} {detected_str:>8}  {mttd_str:>8}"
        )

    lines += [
        "",
        "  Part 2 — Root Cause Analysis",
        "  ------------------------------",
        f"  RCA Accuracy : {pct(rca['rca_accuracy'])}  "
        f"({rca['correct_rca']} / {rca['scenarios_with_detections']} scenarios with detections)",
        "",
        f"  {'Scenario ID':<14} {'Type':<22} {'GT Entity':<14} {'Match':>6}  {'Top Hypothesis'}",
        "  " + "-" * 80,
    ]
    for s in rca["per_scenario_rca"]:
        match_str = "YES" if s["match"] else "no"
        hyp = s["top_hypothesis"] or (s.get("note") or "—")
        gt = s["gt_root_cause"] or "—"
        lines.append(
            f"  {s['scenario_id']:<14} {s['scenario_type']:<22} {gt:<14} {match_str:>6}  {hyp}"
        )

    lines += [
        "",
        "  Part 3 — Decision Performance (simulated)",
        "  ------------------------------------------",
        f"  Mean MTTM    : {sec(dec['mean_mttm_seconds'])}",
        f"  Z3 Pass Rate : {pct(dec['z3_pass_rate'])}",
        f"  Note: {dec['note']}",
        "",
        f"  {'Scenario ID':<14} {'Type':<22} {'Decision':<22} {'Autonomy':<18} {'MTTM':>6}  Z3",
        "  " + "-" * 98,
    ]
    for s in dec["per_scenario_decisions"]:
        dec_str = s["decision"] or "—"
        auto_str = s["autonomy_level"] or "—"
        mttm_str = sec(s["mttm_seconds"]) if s["mttm_seconds"] is not None else "—"
        z3_str = "pass" if s["z3_verified"] else ("fail" if s["z3_verified"] is False else "—")
        lines.append(
            f"  {s['scenario_id']:<14} {s['scenario_type']:<22} {dec_str:<22} {auto_str:<18} {mttm_str:>6}  {z3_str}"
        )

    lines += ["", "=" * 70, ""]
    return "\n".join(lines)


def _save_report(report: dict[str, Any], report_text: str) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    json_path = EVAL_DIR / "report.json"
    txt_path = EVAL_DIR / "report.txt"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(txt_path, "w") as f:
        f.write(report_text)

    print(f"  report.json  →  {json_path}")
    print(f"  report.txt   →  {txt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def evaluate() -> None:
    print("\nCyberCypher 5.0 — Evaluation Pipeline")
    print("=" * 42)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nStep 1/5  Loading data files…")
    train_df, test_df, gt_df, metadata = _load_data()

    test_scenarios = [s for s in metadata["scenarios"] if s["split"] == "test"]
    print(f"          train rows: {len(train_df):,}  |  test rows: {len(test_df):,}")
    print(f"          test scenarios: {len(test_scenarios)}")

    # ── Reconstruct snapshots ──────────────────────────────────────────────
    print("\nStep 2/5  Reconstructing test snapshots from flat CSV…")
    test_snapshots = _df_to_snapshots(test_df)
    print(f"          {len(test_snapshots):,} snapshots reconstructed")

    # Filter ground truth to test period timestamps
    gt_df_test = gt_df.copy()
    gt_df_test["timestamp"] = pd.to_datetime(gt_df_test["timestamp"], utc=True)
    test_start = pd.to_datetime(test_df["timestamp"], utc=True).min()
    gt_df_test = gt_df_test[gt_df_test["timestamp"] >= test_start]

    # ── Part 1: Detection ──────────────────────────────────────────────────
    print("\nStep 3/5  Part 1 — Detection evaluation…")
    detection = _run_detection(train_df, test_snapshots, gt_df_test, test_scenarios)
    print(
        f"          Precision={detection['precision']:.3f}  "
        f"Recall={detection['recall']:.3f}  "
        f"F1={detection['f1']:.3f}  "
        f"MTTD={detection['mean_mttd_seconds']}s"
    )

    # ── Part 2: RCA ────────────────────────────────────────────────────────
    print("\nStep 4/5  Part 2 — Root cause analysis evaluation…")
    rca = _run_rca(
        test_scenarios,
        detection.pop("_detected_anomalies_by_ts"),
    )
    rca_pct = f"{rca['rca_accuracy'] * 100:.1f}%" if rca["rca_accuracy"] is not None else "N/A"
    print(f"          RCA Accuracy={rca_pct}  "
          f"({rca['correct_rca']}/{rca['scenarios_with_detections']} with detections)")

    # ── Part 3: Decision (mock) ────────────────────────────────────────────
    print("\nStep 4/5  Part 3 — Decision performance (simulated)…")
    decision = _run_decision_eval(test_scenarios, detection)

    # ── Part 4: Report ────────────────────────────────────────────────────
    print("\nStep 5/5  Part 4 — Generating report…")
    report = _build_report(detection, rca, decision, metadata)
    report_text = _format_report_text(report)
    _save_report(report, report_text)

    print(report_text)


if __name__ == "__main__":
    evaluate()
