"""Synthetic network incident generator for LLM fine-tuning.

Generates 1000+ labeled network incident scenarios covering:
- Diverse scenario types: congestion, DDoS, fiber cut, BGP anomaly, hardware failure, etc.
- Multiple severity levels: low, medium, high, critical
- Regional diversity: core, aggregation, edge, peering layers
- Compound/multi-fault scenarios
- Partial observability (missing or noisy metrics)

Usage:
    python -m src.models.llm_finetune.synthetic_incident_generator \
        --count 1000 \
        --output data/llm_finetune/synthetic_incidents.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Scenario type definitions
# ---------------------------------------------------------------------------

_SCENARIO_TYPES = [
    "congestion_cascade",
    "ddos_attack",
    "fiber_cut",
    "bgp_anomaly",
    "hardware_failure",
    "config_error",
    "memory_exhaustion",
    "cpu_overload",
    "packet_storm",
    "sla_breach",
    "asymmetric_routing",
    "flapping_link",
    "optical_degradation",
    "firmware_bug",
    "capacity_exhaustion",
]

_SEVERITIES = ["low", "medium", "high", "critical"]

_REGIONS = [
    "core_us_east",
    "core_us_west",
    "core_eu_central",
    "agg_nyc_01",
    "agg_lax_01",
    "agg_ams_01",
    "edge_bos_01",
    "edge_sfo_01",
    "edge_lon_01",
    "peering_nyc_ix",
    "peering_ams_ix",
]

_ACTION_TYPES = [
    "reroute",
    "rate_limit",
    "config_rollback",
    "scale_capacity",
    "null_action",
    "block_source",
    "adjust_bgp_weight",
    "restart_process",
    "failover",
    "increase_buffer",
]

_OUTCOMES = ["effective", "partially_effective", "ineffective", "harmful"]

_SYSTEM_PROMPT = (
    "You are a network operations AI assistant specializing in autonomous ISP "
    "network fault detection, root cause analysis, and remediation. "
    "Analyze the provided network telemetry and anomaly context, then recommend "
    "the most appropriate remediation action with a clear rationale."
)

# ---------------------------------------------------------------------------
# Scenario templates — per scenario type
# ---------------------------------------------------------------------------

_SCENARIO_TEMPLATES: dict[str, dict[str, Any]] = {
    "congestion_cascade": {
        "root_causes": [
            "upstream_link_saturation",
            "traffic_spike_from_cdn",
            "peering_link_failure_reroute",
        ],
        "descriptions": [
            "Progressive congestion spreading from core to edge routers due to traffic burst",
            "Cascading queue buildup after upstream peering link failure triggered reroute",
            "CDN prefetch storm saturating multiple aggregation links simultaneously",
        ],
        "metric_profiles": {
            "utilization_pct": (75.0, 98.0),
            "latency_ms": (30.0, 120.0),
            "packet_loss_pct": (0.5, 4.0),
            "buffer_drops": (200.0, 2000.0),
            "jitter_ms": (5.0, 40.0),
        },
        "recommended_actions": ["reroute", "rate_limit", "adjust_bgp_weight"],
    },
    "ddos_attack": {
        "root_causes": [
            "volumetric_udp_flood",
            "syn_flood_attack",
            "amplification_attack",
        ],
        "descriptions": [
            "Volumetric UDP flood from botnet overwhelming edge router interfaces",
            "SYN flood exhausting connection table on edge router",
            "DNS amplification attack with spoofed source IPs targeting edge",
        ],
        "metric_profiles": {
            "utilization_pct": (85.0, 100.0),
            "latency_ms": (50.0, 500.0),
            "packet_loss_pct": (5.0, 30.0),
            "pps_rate": (500000.0, 5000000.0),
            "flow_count": (10000.0, 200000.0),
        },
        "recommended_actions": ["block_source", "rate_limit", "reroute"],
    },
    "fiber_cut": {
        "root_causes": [
            "physical_cable_damage",
            "connector_failure",
            "optical_transceiver_fault",
        ],
        "descriptions": [
            "Physical fiber cut on backbone link causing complete path loss",
            "Optical connector degradation causing intermittent signal loss",
            "Transceiver failure on core router reducing link capacity to zero",
        ],
        "metric_profiles": {
            "utilization_pct": (0.0, 5.0),  # link is down or near-down
            "latency_ms": (0.0, 1.0),
            "packet_loss_pct": (90.0, 100.0),
            "optical_power_dbm": (-30.0, -15.0),
            "error_rate": (0.1, 1.0),
        },
        "recommended_actions": ["failover", "reroute"],
    },
    "bgp_anomaly": {
        "root_causes": [
            "route_leak",
            "bgp_session_flap",
            "misconfigured_route_policy",
        ],
        "descriptions": [
            "BGP route leak from downstream AS causing suboptimal traffic paths",
            "BGP session flapping due to keepalive timeout causing route churn",
            "Misconfigured route-map redistributing internal prefixes to external peers",
        ],
        "metric_profiles": {
            "utilization_pct": (40.0, 80.0),
            "latency_ms": (20.0, 200.0),
            "packet_loss_pct": (0.1, 3.0),
            "bgp_updates_per_sec": (50.0, 5000.0),
            "route_churn": (10.0, 500.0),
        },
        "recommended_actions": ["config_rollback", "adjust_bgp_weight", "reroute"],
    },
    "hardware_failure": {
        "root_causes": [
            "linecard_failure",
            "power_supply_fault",
            "fan_failure_thermal",
        ],
        "descriptions": [
            "Linecard failure on core router reducing available forwarding capacity",
            "Power supply fault causing partial chassis power loss",
            "Thermal event from fan failure causing CPU throttling and packet drops",
        ],
        "metric_profiles": {
            "utilization_pct": (60.0, 95.0),
            "latency_ms": (15.0, 80.0),
            "packet_loss_pct": (1.0, 15.0),
            "cpu_pct": (70.0, 99.0),
            "temperature_c": (65.0, 95.0),
        },
        "recommended_actions": ["failover", "restart_process", "reroute"],
    },
    "config_error": {
        "root_causes": [
            "acl_misconfiguration",
            "qos_policy_error",
            "mtu_mismatch",
        ],
        "descriptions": [
            "ACL misconfiguration blocking legitimate traffic class",
            "QoS policy error starving high-priority voice/video traffic",
            "MTU mismatch causing excessive fragmentation and latency spikes",
        ],
        "metric_profiles": {
            "utilization_pct": (30.0, 70.0),
            "latency_ms": (25.0, 300.0),
            "packet_loss_pct": (2.0, 20.0),
            "fragmentation_rate": (5.0, 80.0),
            "queued_packets": (1000.0, 50000.0),
        },
        "recommended_actions": ["config_rollback"],
    },
    "memory_exhaustion": {
        "root_causes": [
            "route_table_overflow",
            "memory_leak_process",
            "arp_table_exhaustion",
        ],
        "descriptions": [
            "Route table overflow causing FIB inconsistency and packet drops",
            "Memory leak in routing process consuming available heap",
            "ARP table exhaustion from rapid endpoint churn",
        ],
        "metric_profiles": {
            "utilization_pct": (50.0, 85.0),
            "latency_ms": (10.0, 60.0),
            "packet_loss_pct": (0.5, 8.0),
            "memory_pct": (85.0, 99.0),
            "cpu_pct": (60.0, 90.0),
        },
        "recommended_actions": ["restart_process", "config_rollback"],
    },
    "cpu_overload": {
        "root_causes": [
            "control_plane_storm",
            "logging_overload",
            "snmp_polling_overload",
        ],
        "descriptions": [
            "Control plane CPU saturated by excessive OSPF hello packets",
            "Logging subsystem consuming excessive CPU due to debug storm",
            "SNMP polling storm overwhelming router management CPU",
        ],
        "metric_profiles": {
            "utilization_pct": (40.0, 75.0),
            "latency_ms": (20.0, 100.0),
            "packet_loss_pct": (0.2, 5.0),
            "cpu_pct": (90.0, 100.0),
            "process_count": (200.0, 600.0),
        },
        "recommended_actions": ["restart_process", "rate_limit"],
    },
    "packet_storm": {
        "root_causes": [
            "broadcast_storm_l2",
            "multicast_flood",
            "spanning_tree_loop",
        ],
        "descriptions": [
            "Layer-2 broadcast storm from misconfigured switch flooding uplinks",
            "Multicast flood from PIM-SM configuration error consuming bandwidth",
            "Spanning tree loop causing traffic multiplication across ring topology",
        ],
        "metric_profiles": {
            "utilization_pct": (80.0, 100.0),
            "latency_ms": (40.0, 400.0),
            "packet_loss_pct": (3.0, 25.0),
            "broadcast_pps": (10000.0, 500000.0),
            "buffer_drops": (500.0, 5000.0),
        },
        "recommended_actions": ["rate_limit", "config_rollback", "block_source"],
    },
    "sla_breach": {
        "root_causes": [
            "latency_sla_exceeded",
            "packet_loss_sla_exceeded",
            "availability_sla_breach",
        ],
        "descriptions": [
            "End-to-end latency exceeding SLA threshold for premium customer segment",
            "Packet loss rate above SLA limit for enterprise customer prefix",
            "Link availability dropping below 99.9% SLA due to intermittent faults",
        ],
        "metric_profiles": {
            "utilization_pct": (60.0, 90.0),
            "latency_ms": (50.0, 200.0),
            "packet_loss_pct": (1.0, 10.0),
            "availability_pct": (95.0, 99.8),
            "jitter_ms": (10.0, 50.0),
        },
        "recommended_actions": ["reroute", "scale_capacity", "adjust_bgp_weight"],
    },
    "asymmetric_routing": {
        "root_causes": [
            "unequal_cost_paths",
            "policy_routing_conflict",
            "metric_misconfiguration",
        ],
        "descriptions": [
            "Asymmetric routing causing stateful firewall drops for return traffic",
            "Policy routing conflict sending traffic on suboptimal return path",
            "OSPF metric misconfiguration creating path asymmetry",
        ],
        "metric_profiles": {
            "utilization_pct": (30.0, 65.0),
            "latency_ms": (20.0, 80.0),
            "packet_loss_pct": (1.0, 12.0),
            "rtt_delta_ms": (15.0, 100.0),
            "tcp_retransmit_pct": (3.0, 20.0),
        },
        "recommended_actions": ["config_rollback", "adjust_bgp_weight"],
    },
    "flapping_link": {
        "root_causes": [
            "physical_intermittent_fault",
            "sfp_degradation",
            "keepalive_timeout",
        ],
        "descriptions": [
            "Link flapping due to physical layer intermittent fault causing route churn",
            "SFP module degradation causing periodic optical loss and link reset",
            "Keepalive timeout on WAN link causing periodic session drops",
        ],
        "metric_profiles": {
            "utilization_pct": (20.0, 60.0),
            "latency_ms": (5.0, 50.0),
            "packet_loss_pct": (5.0, 40.0),
            "link_flap_count": (5.0, 100.0),
            "error_rate": (0.01, 0.5),
        },
        "recommended_actions": ["failover", "reroute"],
    },
    "optical_degradation": {
        "root_causes": [
            "fiber_bend_loss",
            "connector_contamination",
            "amplifier_gain_reduction",
        ],
        "descriptions": [
            "Optical power degradation from fiber bend causing intermittent errors",
            "Connector contamination reducing optical SNR below threshold",
            "EDFA amplifier gain reduction affecting long-haul link quality",
        ],
        "metric_profiles": {
            "utilization_pct": (50.0, 80.0),
            "latency_ms": (5.0, 20.0),
            "packet_loss_pct": (0.1, 5.0),
            "optical_power_dbm": (-10.0, -5.0),
            "ber": (1e-8, 1e-4),
        },
        "recommended_actions": ["failover", "reroute"],
    },
    "firmware_bug": {
        "root_causes": [
            "forwarding_plane_bug",
            "memory_corruption",
            "scheduler_deadlock",
        ],
        "descriptions": [
            "Firmware bug in forwarding plane causing periodic traffic blackholing",
            "Memory corruption bug triggered by specific traffic pattern",
            "Scheduler deadlock in firmware causing packet queues to stop draining",
        ],
        "metric_profiles": {
            "utilization_pct": (40.0, 70.0),
            "latency_ms": (10.0, 150.0),
            "packet_loss_pct": (0.5, 15.0),
            "cpu_pct": (50.0, 95.0),
            "error_rate": (0.005, 0.1),
        },
        "recommended_actions": ["config_rollback", "restart_process", "failover"],
    },
    "capacity_exhaustion": {
        "root_causes": [
            "sustained_traffic_growth",
            "backup_traffic_rerouted",
            "peering_capacity_full",
        ],
        "descriptions": [
            "Sustained organic traffic growth exceeding provisioned link capacity",
            "Backup traffic rerouted after primary link failure saturating secondary",
            "Peering link at full capacity due to content provider traffic surge",
        ],
        "metric_profiles": {
            "utilization_pct": (90.0, 100.0),
            "latency_ms": (30.0, 150.0),
            "packet_loss_pct": (1.0, 8.0),
            "throughput_gbps": (95.0, 100.0),
            "queue_depth": (80.0, 100.0),
        },
        "recommended_actions": ["scale_capacity", "reroute", "rate_limit"],
    },
}

# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


@dataclass
class IncidentRecord:
    """A single labeled network incident scenario."""

    scenario_type: str
    severity: str
    region: str
    root_cause: str
    description: str
    metrics: dict[str, float]
    recommended_action: str
    outcome: str
    compound: bool = False
    partial_observability: bool = False
    missing_metrics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SyntheticIncidentGenerator:
    """Generates labeled synthetic network incident scenarios for SFT fine-tuning.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    partial_obs_rate:
        Fraction of incidents with partial observability (missing metrics).
    compound_rate:
        Fraction of incidents that are multi-fault compound scenarios.
    """

    def __init__(
        self,
        seed: int = 42,
        partial_obs_rate: float = 0.15,
        compound_rate: float = 0.10,
    ) -> None:
        self._rng = random.Random(seed)
        self.partial_obs_rate = partial_obs_rate
        self.compound_rate = compound_rate

    def generate(self, count: int = 1000) -> list[IncidentRecord]:
        """Generate ``count`` labeled incident records.

        Parameters
        ----------
        count:
            Total number of incidents to generate.

        Returns
        -------
        List of IncidentRecord instances.
        """
        records: list[IncidentRecord] = []

        # Distribute evenly across scenario types, then shuffle
        per_type = max(1, count // len(_SCENARIO_TYPES))
        remainder = count - per_type * len(_SCENARIO_TYPES)

        for idx, scenario_type in enumerate(_SCENARIO_TYPES):
            n = per_type + (1 if idx < remainder else 0)
            for _ in range(n):
                if self._rng.random() < self.compound_rate:
                    records.append(self._generate_compound())
                else:
                    records.append(self._generate_single(scenario_type))

        self._rng.shuffle(records)
        return records[:count]

    def to_sft_jsonl(self, records: list[IncidentRecord]) -> list[dict[str, Any]]:
        """Convert incident records to conversation-format JSONL dicts.

        Parameters
        ----------
        records:
            List of IncidentRecord instances.

        Returns
        -------
        List of dicts each with 'messages' key (conversation format).
        """
        examples = []
        for record in records:
            example = self._record_to_conversation(record)
            examples.append(example)
        return examples

    def export(self, records: list[IncidentRecord], output_path: str) -> str:
        """Export incident records as SFT conversation JSONL.

        Parameters
        ----------
        records:
            List of IncidentRecord instances.
        output_path:
            File path to write.

        Returns
        -------
        Absolute path of written file.
        """
        examples = self.to_sft_jsonl(records)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for example in examples:
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(
            "Exported {} synthetic incidents to {}", len(examples), output_path
        )
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_single(self, scenario_type: str) -> IncidentRecord:
        """Generate a single-fault incident for the given scenario type."""
        template = _SCENARIO_TEMPLATES[scenario_type]
        severity = self._rng.choice(_SEVERITIES)
        region = self._rng.choice(_REGIONS)
        root_cause = self._rng.choice(template["root_causes"])
        description = self._rng.choice(template["descriptions"])
        recommended_action = self._rng.choice(template["recommended_actions"])
        outcome = self._weighted_outcome(recommended_action)

        metrics = self._sample_metrics(template["metric_profiles"], severity)
        partial_obs = self._rng.random() < self.partial_obs_rate
        missing = self._apply_partial_obs(metrics) if partial_obs else []

        return IncidentRecord(
            scenario_type=scenario_type,
            severity=severity,
            region=region,
            root_cause=root_cause,
            description=description,
            metrics=metrics,
            recommended_action=recommended_action,
            outcome=outcome,
            compound=False,
            partial_observability=partial_obs,
            missing_metrics=missing,
            metadata={"source": "synthetic_v1", "template_version": "1.0"},
        )

    def _generate_compound(self) -> IncidentRecord:
        """Generate a compound multi-fault incident from two scenario types."""
        types = self._rng.sample(_SCENARIO_TYPES, 2)
        primary_type, secondary_type = types[0], types[1]

        primary = _SCENARIO_TEMPLATES[primary_type]
        secondary = _SCENARIO_TEMPLATES[secondary_type]

        severity = "high" if self._rng.random() < 0.5 else "critical"
        region = self._rng.choice(_REGIONS)
        root_cause = (
            f"{self._rng.choice(primary['root_causes'])}"
            f"_plus_{self._rng.choice(secondary['root_causes'])}"
        )
        description = (
            f"Compound fault: {self._rng.choice(primary['descriptions'])} "
            f"compounded by {self._rng.choice(secondary['descriptions']).lower()}"
        )

        # Merge metric profiles — take max bounds for compound stress
        merged_profile: dict[str, tuple[float, float]] = {}
        for profile in (primary["metric_profiles"], secondary["metric_profiles"]):
            for metric, (lo, hi) in profile.items():
                if metric in merged_profile:
                    existing_lo, existing_hi = merged_profile[metric]
                    merged_profile[metric] = (min(existing_lo, lo), max(existing_hi, hi))
                else:
                    merged_profile[metric] = (lo, hi)

        metrics = self._sample_metrics(merged_profile, severity, stress=True)
        partial_obs = self._rng.random() < self.partial_obs_rate * 1.5
        missing = self._apply_partial_obs(metrics) if partial_obs else []

        # Compound incidents are harder — prefer more conservative actions
        all_actions = list(
            set(primary["recommended_actions"] + secondary["recommended_actions"])
        )
        recommended_action = self._rng.choice(all_actions)
        outcome = self._weighted_outcome(recommended_action, compound=True)

        return IncidentRecord(
            scenario_type=f"{primary_type}+{secondary_type}",
            severity=severity,
            region=region,
            root_cause=root_cause,
            description=description,
            metrics=metrics,
            recommended_action=recommended_action,
            outcome=outcome,
            compound=True,
            partial_observability=partial_obs,
            missing_metrics=missing,
            metadata={
                "source": "synthetic_v1",
                "primary_type": primary_type,
                "secondary_type": secondary_type,
                "template_version": "1.0",
            },
        )

    def _sample_metrics(
        self,
        profile: dict[str, tuple[float, float]],
        severity: str,
        stress: bool = False,
    ) -> dict[str, float]:
        """Sample metric values from a profile, scaled by severity."""
        severity_scale = {"low": 0.4, "medium": 0.65, "high": 0.85, "critical": 1.0}
        scale = severity_scale.get(severity, 0.7)
        if stress:
            scale = min(1.0, scale + 0.15)

        metrics: dict[str, float] = {}
        for metric, (lo, hi) in profile.items():
            span = hi - lo
            # Bias toward the upper range proportional to severity
            base = lo + span * scale * self._rng.random()
            noise = span * 0.05 * (self._rng.random() - 0.5)
            value = max(lo, min(hi, base + noise))
            metrics[metric] = round(value, 4)

        # Always include the core ISP metrics
        if "utilization_pct" not in metrics:
            metrics["utilization_pct"] = round(
                self._rng.uniform(30.0, 95.0) * scale, 4
            )
        if "latency_ms" not in metrics:
            metrics["latency_ms"] = round(self._rng.uniform(5.0, 100.0) * scale, 4)
        if "packet_loss_pct" not in metrics:
            metrics["packet_loss_pct"] = round(
                self._rng.uniform(0.0, 5.0) * scale, 4
            )

        return metrics

    def _apply_partial_obs(self, metrics: dict[str, float]) -> list[str]:
        """Remove a subset of metrics to simulate partial observability."""
        removable = [k for k in metrics if k not in ("utilization_pct", "latency_ms")]
        if not removable:
            return []
        n_remove = self._rng.randint(1, max(1, len(removable) // 2))
        to_remove = self._rng.sample(removable, n_remove)
        for key in to_remove:
            del metrics[key]
        return to_remove

    def _weighted_outcome(
        self, action: str, compound: bool = False
    ) -> str:
        """Sample a realistic outcome, biased by action quality."""
        # Better actions for the scenario → more likely to be effective
        if action in ("config_rollback", "failover"):
            weights = [0.55, 0.25, 0.15, 0.05]
        elif action == "null_action":
            weights = [0.05, 0.15, 0.50, 0.30]
        else:
            weights = [0.45, 0.30, 0.20, 0.05]

        if compound:
            # Compound faults are harder to resolve
            weights = [max(0.0, w - 0.10) for w in weights]
            weights[2] += 0.20
            total = sum(weights)
            weights = [w / total for w in weights]

        return self._rng.choices(_OUTCOMES, weights=weights, k=1)[0]

    def _record_to_conversation(self, record: IncidentRecord) -> dict[str, Any]:
        """Convert an IncidentRecord to a conversation-format dict."""
        metric_lines = "\n".join(
            f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
            for k, v in list(record.metrics.items())[:12]
        )

        obs_note = ""
        if record.partial_observability and record.missing_metrics:
            obs_note = (
                f"\nNote: The following metrics are unavailable (sensor/collection gap): "
                f"{', '.join(record.missing_metrics)}."
            )

        compound_note = ""
        if record.compound:
            meta = record.metadata
            compound_note = (
                f"\nCompound fault: primary={meta.get('primary_type', 'unknown')}, "
                f"secondary={meta.get('secondary_type', 'unknown')}."
            )

        user_prompt = (
            f"Scenario: {record.scenario_type}\n"
            f"Severity: {record.severity}\n"
            f"Region: {record.region}\n"
            f"Anomaly description: {record.description}\n"
            f"Root cause hypothesis: {record.root_cause} "
            f"(severity={record.severity})\n"
            f"Current metrics:\n{metric_lines}"
            f"{obs_note}"
            f"{compound_note}\n\n"
            f"What is the recommended remediation action?"
        )

        outcome_note = {
            "effective": "This action was effective and resolved the issue.",
            "partially_effective": "This action partially resolved the issue; further tuning may be needed.",
            "ineffective": "Note: this action was ineffective — a different approach should be considered.",
            "harmful": "Note: this action worsened the situation — avoid it for this fault pattern.",
        }.get(record.outcome, "")

        assistant_response = (
            f"Recommended action: {record.recommended_action}\n"
            f"Rationale: Given {record.scenario_type} in {record.region} with root cause "
            f"'{record.root_cause}' at {record.severity} severity, the "
            f"{record.recommended_action} action addresses the primary fault vector. "
        )
        if record.partial_observability:
            assistant_response += (
                "Note: some metrics are unavailable; this recommendation is based on "
                "available telemetry — monitor closely after execution. "
            )
        if record.compound:
            assistant_response += (
                "This is a compound fault; consider staged remediation and verify "
                "each component fault resolves independently. "
            )
        assistant_response += f"\n{outcome_note}"

        return {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            "metadata": {
                "scenario_type": record.scenario_type,
                "severity": record.severity,
                "region": record.region,
                "root_cause": record.root_cause,
                "recommended_action": record.recommended_action,
                "outcome": record.outcome,
                "compound": record.compound,
                "partial_observability": record.partial_observability,
            },
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for synthetic incident generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic network incidents for LLM fine-tuning"
    )
    parser.add_argument("--count", type=int, default=1000, help="Number of incidents")
    parser.add_argument(
        "--output",
        default="data/llm_finetune/synthetic_incidents.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--partial-obs-rate",
        type=float,
        default=0.15,
        help="Fraction of incidents with partial observability",
    )
    parser.add_argument(
        "--compound-rate",
        type=float,
        default=0.10,
        help="Fraction of compound/multi-fault incidents",
    )
    args = parser.parse_args()

    gen = SyntheticIncidentGenerator(
        seed=args.seed,
        partial_obs_rate=args.partial_obs_rate,
        compound_rate=args.compound_rate,
    )
    records = gen.generate(count=args.count)
    path = gen.export(records, args.output)
    print(f"Generated {len(records)} incidents → {path}")

    # Summary stats
    scenario_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    compound_count = 0
    partial_count = 0
    for r in records:
        scenario_counts[r.scenario_type] = scenario_counts.get(r.scenario_type, 0) + 1
        severity_counts[r.severity] = severity_counts.get(r.severity, 0) + 1
        if r.compound:
            compound_count += 1
        if r.partial_observability:
            partial_count += 1

    print(f"\nSeverity distribution: {severity_counts}")
    print(f"Compound incidents: {compound_count} ({100*compound_count/len(records):.1f}%)")
    print(f"Partial observability: {partial_count} ({100*partial_count/len(records):.1f}%)")
    print(f"Unique scenario types: {len(scenario_counts)}")


if __name__ == "__main__":
    main()
