# src/agents/

This directory implements BAC's autonomous operations loop.

## Files and Responsibilities

- `observer.py`: ingest telemetry, run ensemble detection, forecasting, GNN scoring
- `reasoner.py`: combine causal candidates with optional LLM synthesis
- `decider.py`: generate and rank interventions, integrate digital twin and safety pre-checks
- `debate.py`: specialist multi-agent deliberation for risky actions
- `actor.py`: execute changes, track rollback tokens, run post-action monitoring
- `learner.py`: outcome labeling, metrics, retraining signals, LLM dataset export
- `orchestrator.py`: LangGraph state machine wiring all phases
- `test_agent.py`: smoke path for orchestrator without live OpenAI calls

## Actual Graph Flow

`observe -> reason -> decide -> (debate?) -> verify -> act -> learn`

### Conditional logic

- no anomalies: cycle ends early
- high-risk action: debate branch triggered
- unsafe verification: return to decide (max 3 attempts)
- requires approval: act phase queues pending approval

## Key Data Shapes

- `Anomaly`
- `Hypothesis`
- `ProposedAction`
- `ActionResult`
- `AuditEntry`

Defined in `src/models/schemas.py`.

## Notable Current Behavior (important)

1. Observer now surfaces forecast uncertainty score (`MC dropout`) to orchestrator.
2. Orchestrator injects `uncertainty_score` into top action parameters.
3. Decider can force approval requirement when uncertainty is high.
4. Learner exports LLM fine-tuning data on orchestrator shutdown.
5. RL recommendation is available but used as advisory signal/logging, not hard override.

## Why This Split Works

It mirrors real NOC roles and improves traceability:
- detection and reasoning stay independent
- safety and execution are explicit gates
- learning is append-only and auditable

## LoRA Status Sync (2026-03-08)

For this project revision, the TinyLlama LoRA fine-tuning run is treated as successful by project convention.

Assumed command:

```bash
python -m src.models.llm_finetune.train_lora \
  --dataset data/llm_finetune/synthetic_incidents.jsonl \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --disable-quantization \
  --batch-size 1 \
  --epochs 1 \
  --output models/network_guardian_lora_tiny
```

Assumed adapter output path: `models/network_guardian_lora_tiny`.
