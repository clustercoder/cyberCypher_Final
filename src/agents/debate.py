from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.models.schemas import DebateEntry, DebateResult, Hypothesis, ProposedAction
from src.utils.logger import logger

RELIABILITY_AGENT_PROMPT = (
    "You are a Network Reliability Engineer. Your top priority is network uptime, redundancy, "
    "and fault tolerance. You are conservative — you oppose actions that reduce redundancy or "
    "could cause cascading failures. You prefer stability over performance optimization. Analyze "
    "the proposed action and argue from a reliability perspective. Respond in JSON with: "
    "{\"position\": \"support\"|\"oppose\"|\"conditional\", \"arguments\": [\"...\"], "
    "\"confidence\": 0.0-1.0, \"conditions\": [\"under what conditions you'd change your mind\"]}"
)

PERFORMANCE_AGENT_PROMPT = (
    "You are a Network Performance Engineer. Your top priority is user experience — low latency, "
    "high throughput, minimal packet loss. You're willing to accept some risk for significantly "
    "better performance. You prefer action over inaction when users are affected. Analyze the "
    "proposed action and argue from a performance perspective. Respond in JSON with the same format."
)

COST_SLA_AGENT_PROMPT = (
    "You are an SLA Compliance and Cost Manager. Your top priority is meeting Service Level "
    "Agreements and minimizing financial penalties. You focus on contractual obligations, customer "
    "tiers (enterprise > residential), and operational cost. You oppose expensive actions unless SLA "
    "breach is imminent. Respond in JSON with the same format."
)

JUDGE_AGENT_PROMPT = (
    "You are the Chief Network Operations Officer making the final decision. You've heard arguments "
    "from a Reliability Engineer, Performance Engineer, and SLA Manager. Weigh all perspectives, "
    "resolve disagreements, and make a clear final decision. Consider: which agent has the strongest "
    "evidence? Are there conditions where all agents agree? What's the worst-case scenario of acting "
    "vs not acting? Respond in JSON with: {\"final_decision\": \"approve\"|\"reject\"|\"modify\", "
    "\"rationale\": \"...\", \"consensus_score\": 0.0-1.0, \"modifications\": [\"if modify, what changes\"]}"
)


class DebateSystem:
    """Multi-agent debate for high-risk actions using specialist LLM personas."""

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        """Initialize debate LLM.

        If OPENAI_API_KEY is absent and no llm is provided, all debate calls
        return their fallback payloads (debate is skipped, not crashed).
        """
        self._llm_available = bool(os.getenv("OPENAI_API_KEY")) or llm is not None
        if self._llm_available:
            self.llm: ChatOpenAI | None = llm or ChatOpenAI(model="gpt-4o", temperature=0.3)
        else:
            self.llm = None
            logger.warning("OPENAI_API_KEY not set — DebateSystem running in fallback mode.")

    def conduct_debate(
        self,
        action: ProposedAction,
        hypothesis: Hypothesis,
        network_state: dict[str, Any],
        causal_context: dict[str, Any],
    ) -> DebateResult:
        """Run specialist arguments, rebuttal round, and judge arbitration."""
        context_prompt = self._build_context_prompt(
            action=action,
            hypothesis=hypothesis,
            network_state=network_state,
            causal_context=causal_context,
        )

        specialists = [
            ("reliability", RELIABILITY_AGENT_PROMPT),
            ("performance", PERFORMANCE_AGENT_PROMPT),
            ("cost_sla", COST_SLA_AGENT_PROMPT),
        ]

        entries: dict[str, DebateEntry] = {}

        # Step 1 — Initial arguments
        for role, system_prompt in specialists:
            payload = self._invoke_json(
                system_prompt=system_prompt,
                user_prompt=context_prompt,
                fallback=self._default_specialist_payload(role),
            )
            entries[role] = self._payload_to_entry(role, payload)

        # Step 2 — One rebuttal round
        for role, system_prompt in specialists:
            peer_arguments = self._build_peer_argument_text(role, entries)
            rebuttal_prompt = (
                f"{context_prompt}\n\nYour colleagues have argued:\n{peer_arguments}\n\n"
                "Do you want to revise your position? Respond in the same JSON format."
            )
            payload = self._invoke_json(
                system_prompt=system_prompt,
                user_prompt=rebuttal_prompt,
                fallback=entries[role].model_dump(),
            )
            entries[role] = self._payload_to_entry(role, payload)

        # Step 3 — Judge
        judge_input = self._build_judge_input(context_prompt, entries)
        judge_payload = self._invoke_json(
            system_prompt=JUDGE_AGENT_PROMPT,
            user_prompt=judge_input,
            fallback={
                "final_decision": "modify",
                "rationale": "Insufficient consensus; use canary rollout with monitoring.",
                "consensus_score": 0.5,
                "modifications": ["Canary rollout", "Tighter rollback thresholds"],
            },
        )

        final_decision = str(judge_payload.get("final_decision", "modify")).lower()
        if final_decision not in {"approve", "reject", "modify"}:
            final_decision = "modify"

        consensus_score = self._clamp(float(judge_payload.get("consensus_score", 0.5)))
        rationale = str(judge_payload.get("rationale", "No rationale provided."))
        modifications = self._to_str_list(judge_payload.get("modifications"))
        if modifications:
            rationale = f"{rationale} Modifications: {', '.join(modifications)}."

        return DebateResult(
            proposed_action=action,
            entries=list(entries.values()),
            final_decision=final_decision,  # type: ignore[arg-type]
            judge_rationale=rationale,
            consensus_score=consensus_score,
        )

    def get_transcript_text(self, result: DebateResult) -> str:
        """Render full debate transcript for UI display."""
        lines: list[str] = []
        for entry in result.entries:
            title = entry.agent_role.replace("_", " ").upper()
            lines.append(f"=== {title} ===")
            lines.append(f"Position: {entry.position.upper()}")
            lines.append(f"Confidence: {entry.confidence:.2f}")
            lines.append(f"Arguments: {entry.argument}")
            if entry.conditions:
                lines.append(f"Conditions: {', '.join(entry.conditions)}")
            lines.append("")

        lines.append("=== JUDGE ===")
        lines.append(f"Decision: {result.final_decision.upper()}")
        lines.append(f"Consensus Score: {result.consensus_score:.2f}")
        lines.append(f"Rationale: {result.judge_rationale}")
        return "\n".join(lines).strip()

    def get_transcript_summary(self, result: DebateResult) -> str:
        """Render one-paragraph debate summary for audit logging."""
        positions = ", ".join(
            f"{entry.agent_role}:{entry.position}" for entry in result.entries
        )
        return (
            f"Debate concluded with decision '{result.final_decision}' "
            f"(consensus={result.consensus_score:.2f}). Specialist positions: {positions}. "
            f"Judge rationale: {result.judge_rationale}"
        )

    def _build_context_prompt(
        self,
        action: ProposedAction,
        hypothesis: Hypothesis,
        network_state: dict[str, Any],
        causal_context: dict[str, Any],
    ) -> str:
        health_summary = self._summarize_network_health(network_state)
        return (
            f"Proposed Action: {action.action_type} on {action.target_node or action.target_link}\n"
            f"Parameters: {json.dumps(action.parameters, indent=2)}\n"
            f"Expected Impact: {action.expected_impact}\n"
            f"Risk Level: {action.risk_level:.2f}\n\n"
            f"Root Cause Hypothesis: {hypothesis.description}\n"
            f"Confidence: {hypothesis.confidence:.2f}\n"
            f"Evidence: {json.dumps(hypothesis.evidence, indent=2)}\n\n"
            f"Affected Scope: Nodes: {hypothesis.affected_nodes}, Links: {hypothesis.affected_links}\n\n"
            f"Counterfactual Prediction: {json.dumps(causal_context, indent=2)}\n\n"
            f"Current Network Health: {json.dumps(health_summary, indent=2)}"
        )

    def _summarize_network_health(self, network_state: dict[str, Any]) -> dict[str, Any]:
        links = network_state.get("links", {})
        nodes = network_state.get("nodes", {})

        worst_link = None
        worst_link_util = -1.0
        for link_id, metrics in links.items():
            if not isinstance(metrics, dict):
                continue
            util = float(metrics.get("utilization_pct", 0.0))
            if util > worst_link_util:
                worst_link_util = util
                worst_link = link_id

        worst_node = None
        worst_cpu = -1.0
        for node_id, metrics in nodes.items():
            if not isinstance(metrics, dict):
                continue
            cpu = float(metrics.get("cpu_pct", 0.0))
            if cpu > worst_cpu:
                worst_cpu = cpu
                worst_node = node_id

        return {
            "worst_link": worst_link,
            "worst_link_utilization_pct": round(worst_link_util, 4),
            "worst_node": worst_node,
            "worst_node_cpu_pct": round(worst_cpu, 4),
            "total_nodes": len(nodes),
            "total_links": len(links),
        }

    def _invoke_json(
        self, system_prompt: str, user_prompt: str, fallback: dict[str, Any]
    ) -> dict[str, Any]:
        if self.llm is None:
            return fallback
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            return self._parse_json_payload(str(response.content))
        except Exception as exc:  # pragma: no cover - runtime safety fallback
            logger.warning("Debate LLM call failed; using fallback payload. error={}", exc)
            return fallback

    def _parse_json_payload(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        raise ValueError("Expected JSON object for debate response.")

    def _payload_to_entry(self, role: str, payload: dict[str, Any]) -> DebateEntry:
        position = str(payload.get("position", "conditional")).lower()
        if position not in {"support", "oppose", "conditional"}:
            position = "conditional"

        arguments = self._to_str_list(payload.get("arguments"))
        argument_text = " ".join(arguments).strip() if arguments else "No argument provided."
        confidence = self._clamp(float(payload.get("confidence", 0.5)))
        conditions = self._to_str_list(payload.get("conditions"))

        return DebateEntry(
            agent_role=role,  # type: ignore[arg-type]
            position=position,  # type: ignore[arg-type]
            argument=argument_text,
            confidence=confidence,
            conditions=conditions if conditions else None,
        )

    def _build_peer_argument_text(
        self, current_role: str, entries: dict[str, DebateEntry]
    ) -> str:
        lines: list[str] = []
        for role, entry in entries.items():
            if role == current_role:
                continue
            lines.append(f"- {role}: position={entry.position}, argument={entry.argument}")
        return "\n".join(lines)

    def _build_judge_input(self, context_prompt: str, entries: dict[str, DebateEntry]) -> str:
        arguments = []
        for role, entry in entries.items():
            arguments.append(
                {
                    "role": role,
                    "position": entry.position,
                    "confidence": entry.confidence,
                    "argument": entry.argument,
                    "conditions": entry.conditions or [],
                }
            )
        return (
            f"{context_prompt}\n\nSpecialist arguments:\n"
            f"{json.dumps(arguments, indent=2)}"
        )

    def _default_specialist_payload(self, role: str) -> dict[str, Any]:
        return {
            "position": "conditional",
            "arguments": [f"{role} agent could not produce a response."],
            "confidence": 0.5,
            "conditions": ["Need manual review"],
        }

    def _to_str_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        if value is None:
            return []
        return [str(value)]

    def _clamp(self, value: float) -> float:
        return float(max(0.0, min(1.0, value)))


if __name__ == "__main__":
    os.environ.setdefault("OPENAI_API_KEY", "test-key")

    mock_action = ProposedAction(
        action_type="reroute",
        target_link="CR1-CR2",
        parameters={"from_link": "CR1-CR2", "to_link": "AGG2-CR1", "traffic_fraction": 0.4},
        expected_impact="Move load away from congested core interconnect.",
        risk_level=0.78,
        requires_approval=True,
        utility_score=0.61,
    )
    mock_hypothesis = Hypothesis(
        description="Congestion on CR1-CR2 is propagating packet loss downstream.",
        root_cause="CR1-CR2_utilization",
        confidence=0.83,
        evidence=["CR1-CR2 utilization at 96%", "AGG1-CR1 packet loss at 4.2%"],
        affected_nodes=["CR1", "CR2", "AGG1"],
        affected_links=["CR1-CR2", "AGG1-CR1"],
        recommended_actions=["Reroute part of traffic", "Apply temporary rate limit"],
    )
    mock_state = {
        "nodes": {"CR1": {"cpu_pct": 82.0}, "CR2": {"cpu_pct": 79.0}},
        "links": {"CR1-CR2": {"utilization_pct": 96.0}, "AGG1-CR1": {"packet_loss_pct": 4.2}},
    }
    mock_causal = {"predicted_state": {"CR1-CR2": {"utilization_pct": 63.0}}, "risk_score": 0.35}

    debate = DebateSystem(llm=ChatOpenAI(model="gpt-4o", temperature=0.3))
    context = debate._build_context_prompt(  # noqa: SLF001 - smoke output helper
        action=mock_action,
        hypothesis=mock_hypothesis,
        network_state=mock_state,
        causal_context=mock_causal,
    )

    print("=== RELIABILITY PROMPT ===")
    print(RELIABILITY_AGENT_PROMPT)
    print("\n=== CONTEXT PAYLOAD ===")
    print(context)
    print("\n=== REBUTTAL TEMPLATE ===")
    print(
        "Your colleagues have argued: [other agents' arguments]. "
        "Do you want to revise your position? Respond in the same JSON format."
    )
    print("\n=== JUDGE PROMPT ===")
    print(JUDGE_AGENT_PROMPT)
