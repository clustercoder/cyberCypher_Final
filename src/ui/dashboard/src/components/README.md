# src/ui/dashboard/src/components/

Dashboard panel/component library.

## Components

- `ControlPanel.jsx`: start/stop/inject/kill-switch actions
- `MetricsPanel.jsx`: health + KPI view (MTTD/MTTM/precision/recall/F1)
- `TopologyGraph.jsx`: D3 topology with anomaly emphasis
- `TelemetryCharts.jsx`: time-series metric charts
- `AgentFeed.jsx`: event timeline from orchestrator phases
- `DebateViewer.jsx`: specialist debate transcript and verdict

## UI Control Flow

`App.jsx` is the integration point:
- receives backend events
- normalizes payloads
- passes data/callbacks into these components

## Why Component Split Matches Operations

Each panel answers a separate operator question:
- what control do I have?
- what is unhealthy now?
- where is issue in topology?
- how are metrics trending?
- what is the agent doing?
- why was risky action approved/rejected?

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
