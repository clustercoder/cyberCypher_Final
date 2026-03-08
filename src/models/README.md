# src/models/

This folder contains model implementations and shared data schemas.
It is the core "intelligence toolkit" used by agents.

## Files

- `schemas.py`: Pydantic contracts for all cross-agent data
- `anomaly_detection.py`: Threshold + EWMA + IsolationForest + ensemble logic
- `forecasting.py`: LSTM forecaster + MC dropout uncertainty + fallback forecaster
- `gnn_anomaly.py`: topology-aware GNN anomaly scorer (optional dependency)
- `rl_traffic_engineering.py`: PPO-based traffic engineering helper (optional dependency)
- `llm_finetune/`: offline fine-tuning data/training utilities

## Data Contracts (Most Important)

`schemas.py` defines stable objects like:
- `Anomaly`
- `Hypothesis`
- `ProposedAction`
- `ActionResult`
- `AuditEntry`
- `DebateResult`

These contracts are used everywhere, so schema changes must be deliberate.

## Control Flow

- Detection path: raw metrics -> detector objects -> anomalies
- Forecast path: history window -> forecasts -> predicted anomalies
- Decision support path: optional RL/GNN outputs -> decider context

## Design Choices

1. **Hybrid detection** over single-model dependency.
2. **Forecast uncertainty** available via MC dropout, so policy can escalate uncertain predictions.
3. **Optional heavy dependencies** (torch-geometric, SB3) fail gracefully to keep core loop alive.
4. **Strict schemas** to prevent runtime payload drift.

## For New Contributors

- Add new model outputs as schema-compatible metadata, not ad-hoc dict keys.
- Keep train/inference paths clearly separated.
- Provide fallback behavior for optional model dependencies.
