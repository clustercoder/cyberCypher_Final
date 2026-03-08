# src/simulator/

This folder is the digital twin runtime for network behavior.
It generates realistic telemetry, injects failures, and applies remediation actions.

## Files

- `topology.py`: static network graph and metadata
- `telemetry.py`: per-minute synthetic metrics with diurnal patterns
- `anomaly_injector.py`: labeled scenario overlays for dataset generation
- `engine.py`: real-time async simulation engine with subscriptions
- `digital_twin.py`: action simulation and risk summary helper
- `test_simulator.py`: simulator-focused tests

## Control Flow

1. Topology defines nodes, links, capacities, baselines.
2. Telemetry generator produces each minute snapshot.
3. Engine loop applies:
- scheduled anomaly effects
- persistent action effects (rate limits/capacity changes)
4. Engine publishes snapshots to subscribers (API/orchestrator).

## Data Flow

Input:
- topology graph
- anomaly schedules or immediate injections
- action payloads from actor

Output:
- timestamped snapshot with node and link metrics
- current typed state for downstream components

## Why This Design

- `TelemetryGenerator` is pure generation logic.
- `SimulationEngine` is orchestration and state mutation.
- `AnomalyInjector` is dataset/time-window labeling utility.

This keeps data generation, runtime state management, and labeling concerns separate.

## Beginner Mental Model

Think of this folder as a controllable mini internet.
Everything else in the project is deciding how to respond to what this mini internet does.
