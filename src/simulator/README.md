# src/simulator/

Digital twin runtime for BAC.
It simulates an ISP network and supports anomaly injection + action application.

## Files

- `topology.py`: graph of nodes/links and capacities
- `telemetry.py`: per-minute metric generation with realistic patterns
- `anomaly_injector.py`: labeled scenario overlays for datasets
- `engine.py`: async runtime loop, subscriptions, action application, rollback support
- `digital_twin.py`: fast risk simulation helper used by decider
- `test_simulator.py`: simulator-focused tests

## Runtime Control Flow

1. telemetry step creates baseline snapshot
2. scheduled/active anomalies modify snapshot
3. active action overrides modify snapshot
4. state stored and broadcast to subscribers

## Action Flow

`ActorAgent` -> `SimulationEngine.apply_action()`

Supported action classes include:
- reroute
- rate_limit
- scale_capacity
- rollback (via token)

## Why Separate Telemetry and Engine

- telemetry generator stays deterministic/testable
- engine manages mutable runtime concerns (timing, subscriptions, action overlays)

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
