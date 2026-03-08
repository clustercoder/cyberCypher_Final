# src/

This folder contains all first-party application logic.
Think of it as BAC's full runtime and product code.

## Module Map

- `simulator/`: synthetic ISP digital twin (topology + telemetry + actions)
- `agents/`: observe/reason/decide/act/learn loop
- `models/`: schemas + ML components
- `causal/`: root-cause and counterfactual engine
- `safety/`: Z3 constraints and proofs
- `rag/`: runbook retrieval system
- `api/`: FastAPI + WebSocket transport
- `ui/`: React dashboard
- `evaluation/`: dataset generation and objective scoring
- `data/`: generated datasets
- `utils/`: shared utilities (logging)

## End-to-End Runtime Path

1. simulator creates current network state
2. agents process and decide
3. safety gate verifies decision
4. actor executes on simulator
5. learner logs outcomes
6. api streams state/events to ui

## Why This Layout

This structure separates concerns by responsibility, not by framework:
- easier testing
- lower coupling
- easier fallback handling when optional dependencies are missing

## Contracts Across Modules

All critical payloads use Pydantic models in `models/schemas.py`.
That keeps inter-module interfaces explicit and stable.

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
