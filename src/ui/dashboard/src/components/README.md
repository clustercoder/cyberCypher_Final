# src/ui/dashboard/src/components/

UI component library for the CyberCypher dashboard.
Each component maps to a distinct operational surface.

## Core Components

- `ControlPanel.jsx`: start/stop, inject anomaly, kill switch
- `MetricsPanel.jsx`: MTTD/MTTM, precision/recall/F1, health summary
- `TopologyGraph.jsx`: D3 network graph with anomaly overlays
- `TelemetryCharts.jsx`: Recharts time-series panels
- `AgentFeed.jsx`: chronological phase/event stream
- `DebateViewer.jsx`: high-risk multi-agent debate transcript

## Control Flow

`App.jsx` is the orchestrator for these components.
It passes normalized props and event handlers down to each panel.
Components do not directly call backend unless given callbacks.

## Data Flow by Panel

- ControlPanel: writes actions to backend via callbacks
- MetricsPanel: reads health and aggregate metrics
- TopologyGraph: reads topology + latest telemetry + anomalies
- TelemetryCharts: reads rolling telemetry history
- AgentFeed: reads phase events
- DebateViewer: reads latest debate payload(s)

## Why This Split

Each panel matches a real operator question:
- "Can I control the system?"
- "What is unhealthy?"
- "Where is the problem in topology?"
- "How are metrics trending?"
- "What is the agent doing now?"
- "Why was high-risk action approved or rejected?"

This domain-driven split keeps UI complexity manageable.
