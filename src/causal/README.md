# src/causal/

This folder contains causal reasoning logic used for root-cause analysis and "what-if" simulation.

## Files

- `causal_engine.py`: structural graph + data-learned graph + RCA + counterfactuals

## What The Causal Engine Does

1. Builds **structural edges** from known network mechanics.
- utilization -> packet loss
- utilization -> latency
- CPU -> temperature -> drops
- shared node links can influence each other

2. Learns **data-driven edges** from telemetry.
- primary path uses `pgmpy` structure search (project compatibility requirement)
- fallback path uses lagged correlation

3. Merges both into a combined causal graph.

4. Uses the graph to:
- rank root-cause candidates for anomalies
- estimate impact of hypothetical actions (counterfactual)

## Control Flow

- `build_structural_graph()` during initialization
- `learn_from_data()` when baseline telemetry is available
- `find_root_cause()` during reasoning phase
- `run_counterfactual()` during decision/verification phase

## Why Structural + Learned Together

- Structural edges provide safe, domain-correct priors.
- Learned edges capture patterns not hand-coded.
- Combined graph gives better coverage and better explainability.

## Beginner Mental Model

Treat each metric as a node in a cause graph.
If A can influence B, draw A -> B.
When B is anomalous, walk upstream to find likely origins.
