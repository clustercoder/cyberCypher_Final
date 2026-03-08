from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.models.schemas import ActionResult, Hypothesis, ProposedAction
from src.utils.logger import logger

try:
    from src.models.llm_finetune.dataset_generator import LLMDatasetGenerator
    _LLM_GEN_AVAILABLE = True
except ImportError:
    _LLM_GEN_AVAILABLE = False


class LearnerAgent:
    """Tracks action outcomes and derives policy/model improvement signals."""

    def __init__(self) -> None:
        """Initialize learner memory stores."""
        self.outcome_log: list[dict[str, Any]] = []
        self.success_rates: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"successes": 0, "total": 0}
        )
        self.threshold_adjustments: list[str] = []
        self.training_data: list[dict[str, Any]] = []
        self._mttd_samples: list[float] = []
        self._mttm_samples: list[float] = []
        self._anomaly_detection_data: list[dict[str, Any]] = []
        self._retrain_triggers: list[dict[str, Any]] = []

        # LLM dataset generator (optional)
        self._llm_gen: LLMDatasetGenerator | None = None  # type: ignore[type-arg]
        if _LLM_GEN_AVAILABLE:
            try:
                self._llm_gen = LLMDatasetGenerator()  # type: ignore[assignment]
            except Exception as exc:
                logger.warning("LLMDatasetGenerator init failed: {}", exc)

    def record_outcome(
        self, action: ProposedAction, result: ActionResult, hypothesis: Hypothesis
    ) -> None:
        """Record action result, label outcome quality, and update learned statistics."""
        pre = result.pre_metrics
        post = result.post_metrics
        outcome = self._label_outcome(pre, post)

        scenario_type = self._scenario_type(hypothesis)
        pair_key = (scenario_type, action.action_type)
        self.success_rates[pair_key]["total"] += 1
        if outcome in {"effective", "partially_effective"}:
            self.success_rates[pair_key]["successes"] += 1

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action.model_dump(),
            "pre_metrics": dict(pre),
            "post_metrics": dict(post),
            "outcome": outcome,
            "scenario_type": scenario_type,
            "hypothesis": hypothesis.model_dump(),
        }
        self.outcome_log.append(record)

        training_row = self._build_training_row(action, hypothesis, pre, outcome, scenario_type)
        self.training_data.append(training_row)

        if outcome == "harmful":
            message = (
                f"Harmful outcome detected for action_type={action.action_type} "
                f"scenario={scenario_type}. Recommend lowering future utility weighting."
            )
            logger.warning(message)
            self.threshold_adjustments.append(message)

    def get_success_rate(self, action_type: str, scenario_type: str | None = None) -> float:
        """Return empirical success rate for action type, optionally scoped by scenario."""
        if scenario_type is not None:
            rec = self.success_rates.get((scenario_type, action_type))
            if not rec or rec["total"] == 0:
                return 0.5
            return rec["successes"] / rec["total"]

        total = 0
        successes = 0
        for (scenario, atype), rec in self.success_rates.items():
            if atype != action_type:
                continue
            _ = scenario
            total += rec["total"]
            successes += rec["successes"]
        if total == 0:
            return 0.5
        return successes / total

    def get_recommendations(self) -> list[str]:
        """Analyze historical outcomes and return operational recommendations."""
        recommendations: list[str] = []
        total = len(self.outcome_log)
        if total == 0:
            return recommendations

        harmful_or_ineffective = sum(
            1 for row in self.outcome_log if row["outcome"] in {"ineffective", "harmful"}
        )
        false_positive_rate = harmful_or_ineffective / total
        if false_positive_rate > 0.30:
            recommendations.append("Recommend raising anomaly detection thresholds")

        failures_by_pair: dict[tuple[str, str], tuple[int, int]] = {}
        for pair, rec in self.success_rates.items():
            failures = rec["total"] - rec["successes"]
            failures_by_pair[pair] = (failures, rec["total"])

        for (scenario_type, action_type), (failures, total_actions) in failures_by_pair.items():
            if total_actions >= 3 and failures / total_actions >= 0.6:
                recommendations.append(
                    f"Recommend deprioritizing {action_type} for {scenario_type}"
                )

        if self._mttm_samples:
            avg_mttm = sum(self._mttm_samples) / len(self._mttm_samples)
            if avg_mttm > 600:
                recommendations.append(
                    "Recommend lowering confidence thresholds for auto-execution"
                )

        return recommendations

    def export_training_data(self) -> pd.DataFrame:
        """Export accumulated labeled training examples as a DataFrame."""
        if not self.training_data:
            return pd.DataFrame(
                columns=["anomaly_type", "action_type", "outcome"]
            )
        return pd.DataFrame(self.training_data)

    def get_metrics(self) -> dict[str, Any]:
        """Return aggregate learner metrics."""
        total_actions = len(self.outcome_log)
        outcomes = Counter(row["outcome"] for row in self.outcome_log)
        actions = Counter(row["action"]["action_type"] for row in self.outcome_log)

        successes = outcomes.get("effective", 0) + outcomes.get("partially_effective", 0)
        success_rate = (successes / total_actions) if total_actions else 0.0
        false_positive_rate = (
            (outcomes.get("ineffective", 0) + outcomes.get("harmful", 0)) / total_actions
            if total_actions
            else 0.0
        )

        mttd_avg = sum(self._mttd_samples) / len(self._mttd_samples) if self._mttd_samples else 0.0
        mttm_avg = sum(self._mttm_samples) / len(self._mttm_samples) if self._mttm_samples else 0.0

        return {
            "total_actions": total_actions,
            "success_rate": success_rate,
            "mttd_avg_seconds": mttd_avg,
            "mttm_avg_seconds": mttm_avg,
            "false_positive_rate": false_positive_rate,
            "actions_by_type": dict(actions),
            "outcomes_by_type": dict(outcomes),
        }

    def _label_outcome(self, pre: dict[str, float], post: dict[str, float]) -> str:
        improved = 0
        unchanged = 0
        harmful = False

        for key, pre_val in pre.items():
            if key not in post:
                continue
            post_val = post[key]
            metric = key.split(".", 1)[1] if "." in key else key

            if self._is_harmful(metric, pre_val, post_val):
                harmful = True

            if metric == "latency_ms":
                if pre_val > 0 and (pre_val - post_val) / pre_val > 0.10:
                    improved += 1
                else:
                    unchanged += 1
            elif metric == "utilization_pct":
                if pre_val > 0 and (pre_val - post_val) / pre_val > 0.05:
                    improved += 1
                else:
                    unchanged += 1
            elif metric == "packet_loss_pct":
                if pre_val > 0 and (pre_val - post_val) / pre_val > 0.50:
                    improved += 1
                else:
                    unchanged += 1

        if harmful:
            return "harmful"
        if improved > unchanged and improved > 0:
            return "effective"
        if improved > 0 and unchanged > 0:
            return "partially_effective"
        return "ineffective"

    def _is_harmful(self, metric: str, pre_val: float, post_val: float) -> bool:
        if pre_val <= 0:
            return False

        if metric in {
            "latency_ms",
            "utilization_pct",
            "packet_loss_pct",
            "cpu_pct",
            "memory_pct",
            "temperature_c",
            "buffer_drops",
        }:
            return post_val > pre_val * 1.2

        if metric == "throughput_gbps":
            return post_val < pre_val * 0.8

        return False

    def _build_training_row(
        self,
        action: ProposedAction,
        hypothesis: Hypothesis,
        pre_metrics: dict[str, float],
        outcome: str,
        scenario_type: str,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for key, value in pre_metrics.items():
            row[f"pre_{key}"] = value
        row["anomaly_type"] = scenario_type
        row["action_type"] = action.action_type
        row["hypothesis_confidence"] = hypothesis.confidence
        row["outcome"] = outcome
        return row

    def append_llm_training_example(
        self,
        user_prompt: str,
        assistant_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add a (user, assistant) conversation example to the LLM dataset generator.

        Returns True if the example was added, False if generator unavailable.
        """
        if self._llm_gen is None:
            return False
        self._llm_gen.add_example(user_prompt, assistant_response, metadata)
        return True

    def export_llm_training_data(self, filename: str | None = None) -> str | None:
        """Export accumulated LLM conversation examples to a JSONL file.

        Also drains outcome_log into the generator before exporting.

        Returns path to written file, or None if generator unavailable.
        """
        if self._llm_gen is None:
            return None
        if self.outcome_log:
            added = self._llm_gen.add_from_outcome_log(self.outcome_log)
            logger.info("Added {} outcome-log entries to LLM dataset.", added)
        return self._llm_gen.export(filename=filename)

    def export_anomaly_detection_data(self) -> pd.DataFrame:
        """Export labeled anomaly detection examples for model retraining.

        Returns a DataFrame with columns: entity_id, metric, observed_value,
        expected_value, label (1=anomaly, 0=normal), scenario_type.
        """
        if not self._anomaly_detection_data:
            return pd.DataFrame(
                columns=["entity_id", "metric", "observed_value", "expected_value", "label", "scenario_type"]
            )
        return pd.DataFrame(self._anomaly_detection_data)

    def record_anomaly_detection_sample(
        self,
        entity_id: str,
        metric: str,
        observed_value: float,
        expected_value: float,
        is_anomaly: bool,
        scenario_type: str = "unknown",
    ) -> None:
        """Record a labeled anomaly detection sample for future model retraining."""
        self._anomaly_detection_data.append({
            "entity_id": entity_id,
            "metric": metric,
            "observed_value": observed_value,
            "expected_value": expected_value,
            "label": 1 if is_anomaly else 0,
            "scenario_type": scenario_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def trigger_model_retrain(self, reason: str, model_type: str = "anomaly_detection") -> dict[str, Any]:
        """Log a retrain trigger signal and return a retrain specification.

        Parameters
        ----------
        reason:
            Human-readable reason for triggering retraining.
        model_type:
            Which model to retrain: 'anomaly_detection', 'forecasting', 'llm'.

        Returns
        -------
        Dict with model_type, reason, sample_count, timestamp.
        """
        sample_count = {
            "anomaly_detection": len(self._anomaly_detection_data),
            "forecasting": len(self.training_data),
            "llm": len(self._llm_gen) if self._llm_gen else 0,
        }.get(model_type, 0)

        trigger: dict[str, Any] = {
            "model_type": model_type,
            "reason": reason,
            "sample_count": sample_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._retrain_triggers.append(trigger)
        logger.info(
            "Retrain trigger: model={} reason={} samples={}",
            model_type,
            reason,
            sample_count,
        )
        return trigger

    def get_learning_summary(self) -> dict[str, Any]:
        """Return a comprehensive summary of learning state across all data stores."""
        base_metrics = self.get_metrics()
        return {
            **base_metrics,
            "llm_examples": len(self._llm_gen) if self._llm_gen else 0,
            "anomaly_detection_samples": len(self._anomaly_detection_data),
            "retrain_triggers": len(self._retrain_triggers),
            "threshold_adjustments": len(self.threshold_adjustments),
            "recommendations": self.get_recommendations(),
            "last_retrain_trigger": self._retrain_triggers[-1] if self._retrain_triggers else None,
        }

    def _scenario_type(self, hypothesis: Hypothesis) -> str:
        text = " ".join([hypothesis.root_cause, hypothesis.description, *hypothesis.evidence]).lower()
        if "ddos" in text or "surge" in text:
            return "ddos_surge"
        if "misconfig" in text or "rollback" in text:
            return "misconfiguration"
        if "down" in text or "fiber" in text:
            return "fiber_cut"
        if "hardware" in text or "temperature" in text or "cpu" in text:
            return "hardware_degradation"
        return "congestion_cascade"
