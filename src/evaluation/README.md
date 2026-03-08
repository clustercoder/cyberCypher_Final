# src/evaluation/

Objective performance evaluation tools for BAC.

## Files

- `generate_dataset.py`: builds labeled multi-scenario synthetic dataset
- `evaluate.py`: computes detection and RCA metrics
- `report.json` / `report.txt`: generated summaries

## Evaluation Flow

1. create baseline telemetry
2. inject labeled scenarios
3. split train/test windows
4. train observer on train set
5. run detection on test set
6. evaluate precision/recall/F1/MTTD and RCA correctness

## Why This Folder Is Important

A strong demo is not enough.
This pipeline provides measurable evidence that the system works beyond one curated scenario.

## Common Workflow

```bash
python -m src.evaluation.generate_dataset
python -m src.evaluation.evaluate
```

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
