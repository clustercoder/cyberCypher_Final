# src/ui/dashboard/src/

This is the React application source for the operator dashboard.

## Files

- `App.jsx`: root state container and tab layout
- `main.jsx`: React bootstrap entry
- `index.css`: global theme, primitives, utility styles
- `components/`: all reusable dashboard panels/widgets
- `assets/`: module-imported static assets

## Control Flow

1. `main.jsx` mounts `App`.
2. `App` loads initial REST data and opens WebSocket.
3. WebSocket events update React state stores in `App`.
4. `App` passes normalized props down to components.

## Data Flow

Inbound event types commonly handled:
- `init`
- `telemetry`
- `agent_event`
- `kill_switch`

Derived state in `App` then powers charts, topology, metrics, and debate panels.

## Why Keep Most State In App

A single source of truth avoids inconsistent panel states.
Child components stay mostly presentation-focused and easier to test.
