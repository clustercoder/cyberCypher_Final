# src/ui/dashboard/

Vite + React operator dashboard for BAC.

## Purpose

Provide one place where an operator can:
- observe network state
- inspect agent reasoning/action flow
- intervene when needed

## Runtime UX Flow

1. App fetches initial status/topology via REST
2. App opens WebSocket and ingests real-time events
3. UI state updates across tabs:
- Operations
- Network
- Intelligence

## Build/Run

```bash
cd src/ui/dashboard
npm install
npm run dev
```

Production build:

```bash
npm run build
```

## Environment Variables

- `VITE_API_URL` (REST base URL)
- `VITE_WS_URL` (optional, explicit WS URL)

If not set, local defaults are used for development.

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
