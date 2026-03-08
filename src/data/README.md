# src/data/

Generated datasets used by training and evaluation pipelines.

## Typical Files

- `baseline_telemetry.csv`
- `anomaly_telemetry.csv`
- `train_telemetry.csv`
- `test_telemetry.csv`
- `ground_truth_labels.csv`
- `scenario_metadata.json`

## How Data Moves

1. `src/evaluation/generate_dataset.py` writes these files.
2. `ObserverAgent.train_detectors()` reads baseline/train telemetry.
3. `src/evaluation/evaluate.py` reads test + ground truth to score system quality.

## Why Flat CSV + JSON Metadata

- CSV is easy for pandas and inspection.
- JSON stores richer scenario-level structure cleanly.

## Important Conventions

- timestamps are UTC ISO strings
- entities are split by `entity_type` (`node` or `link`)
- metric column names are shared across simulator, models, and evaluation

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
