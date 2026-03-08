# src/ui/

This folder contains frontend applications.
Right now, the primary UI is the React dashboard in `dashboard/`.

## Role In The System

- visualizes real-time network and agent state
- provides operator controls (start, stop, inject, kill switch)
- surfaces explainability (debates, anomalies, metrics)

## Data Sources

- REST API for baseline/status snapshots
- WebSocket stream for live telemetry and agent events

## Why A Separate UI Layer

Keeping UI independent of backend logic helps:
- iterate visual design quickly
- avoid coupling API internals to rendering code
- support future clients (CLI/mobile) with same backend

Read next: `src/ui/dashboard/README.md`.
