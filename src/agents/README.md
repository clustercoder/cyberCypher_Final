# src/agents/

This folder implements the autonomous decision loop.
Each file maps to one role in the operation lifecycle.

## Files

- `observer.py`: telemetry ingestion + anomaly detection
- `reasoner.py`: causal analysis + optional LLM hypothesis synthesis
- `decider.py`: action generation, utility scoring, policy gates
- `debate.py`: specialist-agent debate for high-risk decisions
- `actor.py`: executes actions with rollback and monitoring
- `learner.py`: tracks outcomes and retraining signals
- `orchestrator.py`: LangGraph state machine wiring all phases
- `test_agent.py`: local smoke/integration-style scenario run

## Control Flow Inside Agents

1. `ObserverAgent.ingest()` stores current snapshot.
2. `ObserverAgent.detect()` emits anomaly objects.
3. `ReasonerAgent.analyze()` turns anomalies into hypotheses.
4. `DeciderAgent.evaluate()` ranks candidate actions.
5. `DebateSystem.conduct_debate()` may run for risky actions.
6. `Z3SafetyVerifier` checks action constraints.
7. `ActorAgent.execute()` applies action to simulation engine.
8. `LearnerAgent.record_outcome()` logs effectiveness.

`AgentOrchestrator` coordinates these steps with a LangGraph state graph.

## Data Flow

Primary models moved between agents:
- `Anomaly`
- `Hypothesis`
- `ProposedAction`
- `ActionResult`
- `AuditEntry`

All are defined in `src/models/schemas.py`.

## Why Separate Agents

The split mirrors real NOC workflows:
- detector and reasoner should not be coupled tightly
- decision policy should stay independent from execution code
- learning should be append-only and resilient

This separation also makes it easier to evaluate each phase in isolation.

## Practical Notes

- Missing `OPENAI_API_KEY` does not hard-fail the full system now; reasoner/debate can degrade.
- Some advanced components (RL/GNN/LLM fine-tune) are optional and may run in fallback mode.
- Keep agent methods side-effect aware. Detection and reasoning should be deterministic where possible; execution and learning naturally mutate state.
