# src/data/

This directory stores generated telemetry datasets used for training and evaluation.

## Typical Files

- `baseline_telemetry.csv`: normal behavior only
- `anomaly_telemetry.csv`: full timeline with injected incidents
- `train_telemetry.csv`: training split
- `test_telemetry.csv`: evaluation split
- `ground_truth_labels.csv`: per-timestamp anomaly labels
- `scenario_metadata.json`: scenario-level metadata and windows

## Data Flow

1. `src/evaluation/generate_dataset.py` writes these files.
2. `ObserverAgent.train_detectors()` consumes train/baseline data.
3. `src/evaluation/evaluate.py` consumes test + ground truth data.

## Why CSV + JSON

- CSV is easy to inspect and use with pandas.
- JSON captures richer scenario metadata that does not fit flat tables cleanly.

## Notes For New Contributors

- Keep timestamps in UTC and ISO format.
- Keep `entity_type` explicit (`node` or `link`).
- Avoid changing column names unless you also update training/evaluation code.
