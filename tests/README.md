# tests/

Automated tests for BAC integration and dataflow integrity.

## Files

- `test_observer_agent.py`: observer-specific behaviors (history, dedupe, forecast, health summary)
- `test_dataflow.py`: stage-by-stage pipeline shape checks
- `test_integration.py`: cross-module integration checks

## Test Philosophy

Most failures in this project happen at boundaries between modules.
So tests prioritize interface contracts and end-to-end object validity.

## Recommended Run Order

```bash
pytest tests/test_observer_agent.py
pytest tests/test_dataflow.py
pytest tests/test_integration.py
```

Run full suite before demos/deployments.

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
