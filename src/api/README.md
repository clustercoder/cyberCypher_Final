# src/api/

This folder exposes CyberCypher to external clients (UI, scripts, tools).
It is the transport layer between autonomous backend logic and human operators.

## Files

- `main.py`: FastAPI app, startup lifecycle, WebSocket broadcast loops

## Control Flow

1. App startup builds:
- `NetworkTopology`
- `SimulationEngine`
- `AgentOrchestrator`
- optional `RAGKnowledgeBase`

2. Background tasks start:
- simulation engine loop
- agent observation loop

3. API endpoints allow operators to:
- start/stop automation
- inject scenarios
- trigger kill switch
- query topology/metrics/status

4. WebSocket channel pushes:
- telemetry snapshots
- anomalies
- agent phase events
- debate/verification/action updates

## Data Flow

Inbound:
- HTTP requests from dashboard controls
- anomaly injection payloads

Outbound:
- JSON REST responses
- real-time WebSocket event stream (`tick`, `telemetry`, `agent_event`, etc.)

## Why This Design

- `lifespan` initialization guarantees the whole stack is ready before serving requests.
- Background loops decouple simulation cadence from HTTP request latency.
- WebSocket broadcasting keeps the UI responsive without polling everything continuously.

## Operational Considerations

- `.env` is loaded at startup so local `uvicorn` runs pick up `OPENAI_API_KEY`.
- Serialization uses JSON-safe payload conversion to avoid datetime/model encoding issues.
- API is intentionally permissive (`CORS *`) for hackathon/demo speed; tighten for production.
