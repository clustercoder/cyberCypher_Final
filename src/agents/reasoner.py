from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.causal.causal_engine import CausalEngine
from src.models.schemas import Anomaly, Hypothesis
from src.simulator.topology import NetworkTopology
from src.utils.logger import logger

SYSTEM_PROMPT = (
    "You are a senior network operations engineer analyzing anomalies in an ISP network. "
    "You form hypotheses about root causes based on causal analysis and network telemetry. "
    "Always respond in valid JSON format."
)


class ReasonerAgent:
    """Combines causal inference and LLM reasoning to form root-cause hypotheses."""

    def __init__(
        self,
        causal_engine: CausalEngine,
        topology: NetworkTopology,
        rag_kb: Optional[Any] = None,
        llm: Optional[ChatOpenAI] = None,
    ) -> None:
        """Initialize reasoner with causal engine and optional LLM + RAG KB.

        If OPENAI_API_KEY is absent and no llm is provided, the reasoner operates
        in causal-only mode (no LLM synthesis; returns causal hypotheses directly).
        """
        self._llm_available = bool(os.getenv("OPENAI_API_KEY")) or llm is not None
        if self._llm_available:
            self.llm: Optional[ChatOpenAI] = llm or ChatOpenAI(model="gpt-4o", temperature=0.2)
        else:
            self.llm = None
            logger.warning("OPENAI_API_KEY not set — ReasonerAgent running in causal-only mode.")
        self.causal_engine = causal_engine
        self.topology = topology
        self.rag_kb = rag_kb

    def analyze(self, anomalies: list[Anomaly], network_state: dict[str, Any]) -> list[Hypothesis]:
        """Analyze anomalies and synthesize ranked hypotheses."""
        if not anomalies:
            return []

        causal_hypotheses = self.causal_engine.find_root_cause(anomalies)
        context = self._query_rag_context(anomalies)
        messages = self._build_analysis_messages(
            anomalies=anomalies,
            causal_hypotheses=causal_hypotheses,
            network_state=network_state,
            rag_context=context,
        )

        if self.llm is not None:
            try:
                response = self.llm.invoke(messages)
                payload = self._parse_json_payload(str(response.content))
                hypotheses = self._payload_to_hypotheses(payload, fallback=causal_hypotheses)
            except Exception as exc:  # pragma: no cover - runtime safety fallback
                logger.warning("LLM synthesis failed; falling back to causal hypotheses. error={}", exc)
                hypotheses = self._fallback_hypotheses(causal_hypotheses)
        else:
            hypotheses = self._fallback_hypotheses(causal_hypotheses)

        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:3]

    def explain(self, hypothesis: Hypothesis) -> str:
        """Return a concise operator-facing explanation for a given hypothesis."""
        prompt = (
            "Explain this network issue diagnosis to a network operator in 2-3 clear sentences: "
            f"{hypothesis.model_dump_json()}"
        )
        try:
            if self.llm is None:
                raise RuntimeError("LLM not available")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return str(response.content).strip()
        except Exception as exc:  # pragma: no cover - runtime safety fallback
            logger.warning("LLM explanation failed. error={}", exc)
            return (
                f"Likely root cause: {hypothesis.root_cause}. "
                f"Confidence is {hypothesis.confidence:.2f}. "
                "Review the evidence list and recommended actions."
            )

    def _query_rag_context(self, anomalies: list[Anomaly]) -> str:
        if self.rag_kb is None:
            return ""

        try:
            metric = anomalies[0].metric_name
            query = f"troubleshooting {metric} issues"
            result = self.rag_kb.query(query)
            if isinstance(result, str):
                return result
            if isinstance(result, list):
                return "\n".join(str(item) for item in result)
            return str(result)
        except Exception as exc:  # pragma: no cover - runtime safety fallback
            logger.warning("RAG query failed. Continuing without RAG context. error={}", exc)
            return ""

    def _build_analysis_messages(
        self,
        anomalies: list[Anomaly],
        causal_hypotheses: list[Hypothesis],
        network_state: dict[str, Any],
        rag_context: str,
    ) -> list[SystemMessage | HumanMessage]:
        anomaly_payload = [
            {
                "metric": a.metric_name,
                "value": a.observed_value,
                "entity": a.node_id or a.link_id,
                "severity": a.severity,
                "confidence": a.confidence,
            }
            for a in anomalies
        ]
        causal_payload = [
            {
                "root_cause": h.root_cause,
                "causal_strength": h.confidence,
                "evidence": h.evidence,
            }
            for h in causal_hypotheses
        ]
        state_summary = self._summarize_network_state(network_state, anomalies)
        topology_context = self._build_topology_context(anomalies)
        operational_knowledge = rag_context if rag_context else "none available"

        user_prompt = (
            "Network anomalies detected:\n"
            f"{json.dumps(anomaly_payload, indent=2)}\n\n"
            "Causal analysis candidates:\n"
            f"{json.dumps(causal_payload, indent=2)}\n\n"
            "Current network state summary:\n"
            f"{state_summary}\n\n"
            "Relevant operational knowledge:\n"
            f"{operational_knowledge}\n\n"
            "Network topology context:\n"
            f"{topology_context}\n\n"
            "Produce a JSON array of hypotheses, each with:\n"
            "- description: one sentence explaining the hypothesis\n"
            "- root_cause: specific node or link causing the issue\n"
            "- confidence: 0-1 score\n"
            "- evidence: list of supporting evidence strings\n"
            "- affected_nodes: list of node IDs affected\n"
            "- affected_links: list of link IDs affected\n"
            "- recommended_actions: list of suggested interventions\n\n"
            "Rank by confidence. Maximum 3 hypotheses."
        )

        return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]

    def _summarize_network_state(
        self, network_state: dict[str, Any], anomalies: list[Anomaly]
    ) -> dict[str, Any]:
        nodes = network_state.get("nodes", {})
        links = network_state.get("links", {})

        affected_nodes = sorted({a.node_id for a in anomalies if a.node_id})
        affected_links = sorted({a.link_id for a in anomalies if a.link_id})

        node_metrics = {
            node_id: nodes.get(node_id, {})
            for node_id in affected_nodes
            if node_id in nodes
        }
        link_metrics = {
            link_id: links.get(link_id, {})
            for link_id in affected_links
            if link_id in links
        }

        return {
            "timestamp": network_state.get("timestamp"),
            "affected_nodes": node_metrics,
            "affected_links": link_metrics,
        }

    def _build_topology_context(self, anomalies: list[Anomaly]) -> dict[str, list[str]]:
        neighbors: dict[str, list[str]] = {}
        graph = self.topology.get_graph()

        for anomaly in anomalies:
            if anomaly.node_id:
                node_id = anomaly.node_id
                if node_id in graph:
                    neighbors[node_id] = sorted(self.topology.get_neighbors(node_id))
                continue

            if anomaly.link_id:
                parts = anomaly.link_id.split("-")
                if len(parts) >= 2:
                    left, right = parts[0], parts[1]
                    if left in graph:
                        neighbors[left] = sorted(self.topology.get_neighbors(left))
                    if right in graph:
                        neighbors[right] = sorted(self.topology.get_neighbors(right))

        return neighbors

    def _payload_to_hypotheses(
        self, payload: Any, fallback: list[Hypothesis]
    ) -> list[Hypothesis]:
        if isinstance(payload, dict):
            if isinstance(payload.get("hypotheses"), list):
                payload = payload["hypotheses"]
            else:
                payload = [payload]

        if not isinstance(payload, list):
            return self._fallback_hypotheses(fallback)

        hypotheses: list[Hypothesis] = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            try:
                confidence = float(item.get("confidence", 0.5))
                confidence = float(max(0.0, min(1.0, confidence)))
                hypothesis = Hypothesis(
                    description=str(item.get("description", "Unknown network issue")),
                    root_cause=str(item.get("root_cause", "unknown")),
                    confidence=confidence,
                    evidence=self._to_str_list(item.get("evidence")),
                    affected_nodes=self._to_str_list(item.get("affected_nodes")),
                    affected_links=self._to_str_list(item.get("affected_links")),
                    recommended_actions=self._to_str_list(item.get("recommended_actions")),
                )
                hypotheses.append(hypothesis)
            except Exception:
                continue

        if not hypotheses:
            return self._fallback_hypotheses(fallback)
        return hypotheses

    def _fallback_hypotheses(self, causal_hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        return [h.model_copy(deep=True) for h in causal_hypotheses[:3]]

    def _to_str_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        if value is None:
            return []
        return [str(value)]

    def _parse_json_payload(self, content: str) -> Any:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end > start:
                return json.loads(text[start : end + 1])
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise


if __name__ == "__main__":
    class _NoOpLLM:
        def invoke(self, _messages: list[Any]) -> Any:  # pragma: no cover - smoke helper
            raise RuntimeError("No network call in smoke mode.")

    os.environ.setdefault("OPENAI_API_KEY", "test-key")

    from src.simulator.topology import NetworkTopology

    topology = NetworkTopology()
    causal_engine = CausalEngine(topology)
    reasoner = ReasonerAgent(causal_engine=causal_engine, topology=topology, llm=_NoOpLLM())  # type: ignore[arg-type]

    now = datetime.now(timezone.utc)
    mock_anomalies = [
        Anomaly(
            timestamp=now,
            metric_name="utilization_pct",
            link_id="CR1-CR2",
            observed_value=95.0,
            expected_value=60.0,
            severity="high",
            confidence=0.9,
            detector_type="threshold",
        ),
        Anomaly(
            timestamp=now,
            metric_name="packet_loss_pct",
            link_id="CR1-CR2",
            observed_value=3.0,
            expected_value=0.1,
            severity="high",
            confidence=0.84,
            detector_type="ewma",
        ),
    ]
    mock_state = {
        "timestamp": now.isoformat(),
        "nodes": {"CR1": {"cpu_pct": 82.0, "memory_pct": 71.0, "temperature_c": 58.0}},
        "links": {"CR1-CR2": {"utilization_pct": 95.0, "latency_ms": 20.0, "packet_loss_pct": 3.0}},
    }
    causal_hypotheses = causal_engine.find_root_cause(mock_anomalies)
    messages = reasoner._build_analysis_messages(  # noqa: SLF001 - debug smoke output
        anomalies=mock_anomalies,
        causal_hypotheses=causal_hypotheses,
        network_state=mock_state,
        rag_context="none available",
    )

    print("=== SYSTEM MESSAGE ===")
    print(messages[0].content)
    print("\n=== USER MESSAGE ===")
    print(messages[1].content)
