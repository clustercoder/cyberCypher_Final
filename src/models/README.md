# src/models/

Core data models and ML components used by BAC.

## Files

- `schemas.py`: Pydantic contracts used across all modules
- `anomaly_detection.py`: threshold + EWMA + IsolationForest + ensemble
- `forecasting.py`: LSTM forecasting + MC dropout uncertainty + fallback forecaster
- `gnn_anomaly.py`: optional topology-aware anomaly scoring (torch-geometric)
- `rl_traffic_engineering.py`: optional PPO-based traffic engineering helper
- `llm_finetune/`: offline dataset generation and LoRA training scripts

## Data Contract Role (`schemas.py`)

`Anomaly`, `Hypothesis`, `ProposedAction`, `ActionResult`, `AuditEntry`, `DebateResult` are shared everywhere.
Changing these contracts impacts:
- orchestrator
- API serialization
- UI rendering
- tests

## Detection + Forecast Strategy

- **ensemble** handles immediate anomalies robustly
- **forecasting** predicts near-future congestion
- **uncertainty score** can escalate decisions to human review

## Optional Model Paths

- GNN: used if torch-geometric is available, otherwise skipped/fallback
- RL: advisory path; policy can load from saved artifact, otherwise heuristic fallback
- Fine-tune scripts: offline only, not in API startup path

## Why Hybrid Modeling

Operational reliability is usually better with:
- interpretable baseline methods
- optional advanced models as additive signals
- graceful degradation when heavy dependencies are missing

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
