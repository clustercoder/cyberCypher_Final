"""Convert agent outcome logs to SFT conversation-format JSONL.

Reads from LearnerAgent training_data and outcome_log, then emits JSONL where
each line is a dict with 'messages': [{'role': 'user', 'content': ...},
{'role': 'assistant', 'content': ...}].
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from src.utils.logger import logger

_SYSTEM_PROMPT = (
    "You are a network operations AI assistant specializing in autonomous ISP "
    "network fault detection, root cause analysis, and remediation. "
    "Analyze the provided network telemetry and anomaly context, then recommend "
    "the most appropriate remediation action with a clear rationale."
)


class LLMDatasetGenerator:
    """Converts agent logs to conversation-format JSONL for SFT fine-tuning.

    Parameters
    ----------
    output_dir:
        Directory where JSONL dataset files are written.
    """

    def __init__(self, output_dir: str = "data/llm_finetune") -> None:
        self.output_dir = output_dir
        self._examples: list[dict[str, Any]] = []

    def add_from_outcome_log(self, outcome_log: list[dict[str, Any]]) -> int:
        """Convert LearnerAgent outcome_log entries to conversation examples.

        Parameters
        ----------
        outcome_log:
            List of outcome records from LearnerAgent.outcome_log.

        Returns
        -------
        Number of examples added.
        """
        added = 0
        for record in outcome_log:
            example = self._outcome_to_example(record)
            if example is not None:
                self._examples.append(example)
                added += 1
        return added

    def add_example(
        self,
        user_prompt: str,
        assistant_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Directly add a (user, assistant) conversation example."""
        example: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ]
        }
        if metadata:
            example["metadata"] = metadata
        self._examples.append(example)

    def export(self, filename: str | None = None) -> str:
        """Write collected examples to a JSONL file.

        Parameters
        ----------
        filename:
            Output filename (auto-generated if None).

        Returns
        -------
        Absolute path of the written file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"sft_dataset_{ts}.jsonl"

        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            for example in self._examples:
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")

        logger.info("Exported {} SFT examples to {}", len(self._examples), path)
        return path

    def __len__(self) -> int:
        return len(self._examples)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _outcome_to_example(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a single outcome_log record to a conversation example."""
        action = record.get("action", {})
        hypothesis = record.get("hypothesis", {})
        pre_metrics = record.get("pre_metrics", {})
        outcome = record.get("outcome", "unknown")
        scenario_type = record.get("scenario_type", "unknown")

        action_type = action.get("action_type", "unknown")
        root_cause = hypothesis.get("root_cause", "unknown")
        description = hypothesis.get("description", "")
        confidence = hypothesis.get("confidence", 0.0)

        if not action_type or not root_cause:
            return None

        # Build user prompt from telemetry context
        metric_lines = "\n".join(
            f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}"
            for k, v in list(pre_metrics.items())[:12]
        )
        user_prompt = (
            f"Scenario: {scenario_type}\n"
            f"Anomaly description: {description}\n"
            f"Root cause hypothesis: {root_cause} (confidence={confidence:.2f})\n"
            f"Pre-action metrics:\n{metric_lines}\n\n"
            f"What is the recommended remediation action?"
        )

        # Build assistant response from action and outcome
        outcome_note = {
            "effective": "This action was effective and resolved the issue.",
            "partially_effective": "This action partially resolved the issue.",
            "ineffective": "Note: this action was ineffective — a different approach may be needed.",
            "harmful": "Note: this action was harmful — avoid it in similar scenarios.",
        }.get(outcome, "")

        assistant_response = (
            f"Recommended action: {action_type}\n"
            f"Rationale: Given {scenario_type} with root cause '{root_cause}', "
            f"the {action_type} action addresses the primary congestion/fault vector. "
            f"Confidence: {confidence:.2f}.\n"
            f"{outcome_note}"
        )

        return {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            "metadata": {
                "outcome": outcome,
                "scenario_type": scenario_type,
                "action_type": action_type,
            },
        }
