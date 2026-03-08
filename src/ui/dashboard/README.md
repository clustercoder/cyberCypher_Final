# src/ui/dashboard/

This is the Vite + React operator dashboard for CyberCypher.
It is designed to show network health and autonomous agent decisions in real time.

## Files (Top Level)

- `src/`: React application code
- `public/`: static assets
- `index.html`: Vite entry HTML
- `package.json`: frontend dependencies/scripts
- `vite.config.js`: Vite build/dev config
- `tailwind.config.js`: Tailwind configuration
- `eslint.config.js`: lint rules

## Runtime Flow

1. App boots and fetches initial status/topology via REST.
2. App opens WebSocket connection.
3. Incoming events update local UI state:
- telemetry history
- active anomalies
- agent phase feed
- debate artifacts

4. Operator actions call REST endpoints:
- start/stop loop
- inject scenario
- kill switch

## UI Design Rationale

Recent polish decisions in this repo favor:
- low visual noise
- consistent spacing/typography
- tabbed layout to reduce clutter
- readable severity/health cues over flashy effects

## Where To Read Next

- `src/ui/dashboard/src/README.md`
- `src/ui/dashboard/src/components/README.md`
