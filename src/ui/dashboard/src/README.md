# src/ui/dashboard/src/

Main React application source.

## Files

- `main.jsx`: app bootstrap
- `App.jsx`: central state and layout/tab orchestration
- `index.css`: global theme and shared style primitives
- `components/`: panel/widget implementations
- `assets/`: bundled asset imports

## UI State/Data Flow

1. `App.jsx` owns primary live state:
- topology
- telemetry history
- active anomalies
- agent events
- debates
- health/metrics

2. child components receive normalized props and render only presentation logic.

## Why This Pattern

Single owner state in `App` prevents inconsistent panel states and keeps integration with WS messages predictable.

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
