# tests/

This folder contains automated tests for integration and dataflow sanity.

## Files

- `test_observer_agent.py`: observer behavior and detection checks
- `test_dataflow.py`: payload/data movement validations
- `test_integration.py`: broader multi-module interaction checks

## Testing Philosophy

The project emphasizes integration confidence because most bugs occur at boundaries between modules.

## What To Verify First

1. observer detects and deduplicates anomalies correctly
2. orchestrator state transitions stay valid
3. API and UI payload shapes remain compatible
4. safety verification blocks unsafe actions

## Suggested Workflow

- run targeted tests after each subsystem change
- run integration tests before demo/release
- keep tests aligned with schema contracts
