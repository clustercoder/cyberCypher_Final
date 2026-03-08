# src/ui/

Frontend layer for BAC.
Currently the active client is the React dashboard in `dashboard/`.

## Role In System

- displays live topology and telemetry
- shows anomaly and agent-phase events
- allows operator control actions
- surfaces debate and safety context

## Data Sources

- REST API for initial snapshots and commands
- WebSocket for live streaming updates

## Why Separate UI Module

Decoupling UI from backend logic improves:
- iteration speed
- testing clarity
- ability to add future clients without backend rewrites

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
