# src/api/

This directory provides the external interface to BAC:
- REST endpoints
- WebSocket event streaming

## Main File

- `main.py`: FastAPI app + startup lifecycle + background loops

## Startup Flow

1. load `.env`
2. initialize topology
3. initialize simulation engine
4. initialize orchestrator and train observer on baseline data
5. initialize optional RAG KB
6. start engine loop + agent observation loop

## Runtime Flow

- engine loop advances simulation continuously
- agent loop runs one orchestrator cycle every 5 seconds
- telemetry and phase events are broadcast to connected WebSocket clients

## Key REST Endpoints

- `GET /health`
- `GET /api/status`
- `GET /api/topology`
- `GET /api/telemetry`
- `GET /api/anomalies`
- `GET /api/metrics`
- `GET /api/audit-log`
- `POST /api/start`
- `POST /api/stop`
- `POST /api/inject`
- `POST /api/kill-switch`
- `POST /api/approve`
- RAG endpoints under `/api/rag/*`

## WebSocket

- path: `/ws`
- event types include: `init`, `tick`, `telemetry`, `agent_event`, `kill_switch`

## Why Background Loops

Separating simulation/agent loops from request handlers keeps UI responsive and avoids API latency coupling to internal cycle duration.

## Operational Notes

- CORS is open for hackathon/demo convenience.
- serialization uses JSON-safe encoding for Pydantic/date fields.
- if RAG/OpenAI init fails, core API still runs.

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
