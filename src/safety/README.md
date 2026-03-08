# src/safety/

Formal safety guardrails for autonomous actions.

## Main File

- `z3_verifier.py`

## What Is Verified

Before execution, candidate action is checked against constraints like:
- max link utilization after action
- no cascading overload on healthy links
- customer impact cap
- action rate-limit cap
- blast radius cap
- rollback path availability

## Verification Flow

1. decider creates proposed action
2. causal/digital twin produce predicted post-action context
3. verifier encodes constraints in Z3
4. returns SAT/UNSAT-style verdict with details and proof text

## Why This Matters

Model confidence is probabilistic.
Safety constraints represent hard requirements that must hold regardless of confidence.

## Extension Pattern

You can register additional constraints without changing existing verification pipeline.

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
