# src/evaluation/

This folder evaluates whether CyberCypher is effective, not just functional.

## Files

- `generate_dataset.py`: creates labeled train/test telemetry dataset
- `evaluate.py`: computes detection and RCA metrics
- `report.json` / `report.txt`: saved evaluation outputs

## Control Flow

1. Generate 24-hour synthetic baseline.
2. Inject planned anomaly scenarios with labels.
3. Split into train/test windows.
4. Train observer on train data.
5. Run detection/RCA on test data.
6. Compute metrics (precision, recall, F1, MTTD, etc.).

## Data Flow

Input:
- topology + telemetry/anomaly injectors

Intermediate:
- CSV telemetry snapshots
- scenario metadata + ground-truth labels

Output:
- quantitative scores and per-scenario diagnostics

## Why This Matters

Hackathon demos can look good visually but still fail objectively.
This folder provides measurable evidence of system quality and helps prevent overfitting to one scenario.

## Beginner Tip

If you add a new detector or policy, run evaluation before and after.
Only keep the change if metrics improve or tradeoffs are clearly justified.
