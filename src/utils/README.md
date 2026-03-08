# src/utils/

Shared utility layer.

## Current File

- `logger.py`: Loguru setup for console + rotating file logs

## Logging Behavior

- colored concise stderr logs for live debugging
- daily rotating file logs in `logs/`

## Why Central Logger Config

A common log format across modules makes distributed debugging much easier for asynchronous systems like BAC.

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
