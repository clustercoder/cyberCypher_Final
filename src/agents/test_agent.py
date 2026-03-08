from __future__ import annotations

import asyncio
import os
from typing import Any

import pandas as pd

from src.agents.orchestrator import AgentOrchestrator
from src.models.schemas import DebateEntry, DebateResult, Hypothesis
from src.simulator.engine import SimulationEngine
from src.simulator.telemetry import TelemetryGenerator
from src.simulator.topology import NetworkTopology


def _build_baselines(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    metric_cols = [
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
        values: dict[str, float] = {}
        for metric in metric_cols:
            if metric not in group.columns:
                continue
            series = pd.to_numeric(group[metric], errors="coerce").dropna()
            if not series.empty:
                values[metric] = float(series.mean())
        baselines[str(entity_id)] = values
    return baselines


async def main() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "test-key")

    # 1. Create topology and simulation engine.
    topology = NetworkTopology()
    engine = SimulationEngine(topology, speed_multiplier=120.0, seed=42)

    # 2. Generate 4h baseline and train observer.
    telemetry = TelemetryGenerator(topology, seed=42)
    baseline_df = telemetry.generate_baseline(duration_hours=4)
    baselines = _build_baselines(baseline_df)

    # 3/4. Build orchestrator (contains causal engine + observer).
    orchestrator = AgentOrchestrator(topology, engine)
    orchestrator.observer.baselines = baselines
    orchestrator.observer.train_detectors(baseline_df)
    orchestrator.causal_engine.learn_from_data(baseline_df)

    # 6. Mock LLM responses so no OpenAI network calls happen.
    def mock_analyze(anomalies: list[Any], _network_state: dict[str, Any]) -> list[Hypothesis]:
        if not anomalies:
            return []
        anomaly = anomalies[0]
        entity = anomaly.link_id or anomaly.node_id or "unknown"
        return [
            Hypothesis(
                description=f"Likely congestion or instability around {entity}.",
                root_cause=f"{entity}_{anomaly.metric_name}",
                confidence=0.82,
                evidence=[f"{anomaly.metric_name} observed at {anomaly.observed_value:.2f}"],
                affected_nodes=[anomaly.node_id] if anomaly.node_id else [],
                affected_links=[anomaly.link_id] if anomaly.link_id else [],
                recommended_actions=["Apply targeted mitigation", "Increase monitoring"],
            )
        ]

    def mock_debate(
        action: Any, hypothesis: Any, _network_state: dict[str, Any], _causal_context: dict[str, Any]
    ) -> DebateResult:
        entries = [
            DebateEntry(agent_role="reliability", position="conditional", argument="Proceed with rollback guard.", confidence=0.74),
            DebateEntry(agent_role="performance", position="support", argument="Users benefit from rapid mitigation.", confidence=0.81),
            DebateEntry(agent_role="cost_sla", position="conditional", argument="Proceed if enterprise SLA risk is high.", confidence=0.69),
        ]
        return DebateResult(
            proposed_action=action,
            entries=entries,
            final_decision="approve",
            judge_rationale=f"Approved for controlled execution based on {hypothesis.root_cause}.",
            consensus_score=0.78,
        )

    orchestrator.reasoner.analyze = mock_analyze  # type: ignore[method-assign]
    orchestrator.debate_system.conduct_debate = mock_debate  # type: ignore[method-assign]

    # 5. Start engine at 120x speed.
    await engine.start()
    await asyncio.sleep(1.0)
    engine.inject_anomaly_now("congestion_cascade", "CR1-CR2", duration_minutes=30)

    # 6/7. Run 3 cycles and print observations.
    for cycle in range(1, 4):
        state = await orchestrator.run_loop(max_cycles=1)
        anomalies = state.get("anomalies", [])
        hypotheses = state.get("hypotheses", [])
        actions = state.get("proposed_actions", [])

        print(f"\n--- Cycle {cycle} ---")
        print(f"Observed anomalies: {len(anomalies)}")
        if anomalies:
            first = anomalies[0]
            print(
                f"  Top anomaly: metric={first.get('metric_name')} "
                f"entity={first.get('node_id') or first.get('link_id')} "
                f"severity={first.get('severity')} confidence={first.get('confidence')}"
            )

        print(f"Hypotheses: {len(hypotheses)}")
        if hypotheses:
            print(
                f"  Top hypothesis: root_cause={hypotheses[0].get('root_cause')} "
                f"confidence={hypotheses[0].get('confidence')}"
            )

        print(f"Actions proposed: {len(actions)}")
        if actions:
            print(
                f"  Top action: type={actions[0].get('action_type')} "
                f"utility={actions[0].get('utility_score')} "
                f"z3_verified={actions[0].get('z3_verified')}"
            )

    orchestrator.stop()
    await engine.stop()
    print("\n✅ Phase 3 Complete — Agent core operational")


if __name__ == "__main__":
    asyncio.run(main())
