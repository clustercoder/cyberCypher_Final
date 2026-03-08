import { useCallback, useEffect, useRef, useState } from 'react'
import ControlPanel from './components/ControlPanel'
import TopologyGraph from './components/TopologyGraph'
import TelemetryCharts from './components/TelemetryCharts'
import AgentFeed from './components/AgentFeed'
import MetricsPanel from './components/MetricsPanel'
import DebateViewer from './components/DebateViewer'
import MagicCard from './components/magicui/MagicCard'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'

const MAX_HISTORY = 60
const MAX_EVENTS = 200
const MAX_DEBATES = 24
const TABS = [
  { id: 'operations', label: 'Operations', hint: 'Control, health, anomalies' },
  { id: 'network', label: 'Network', hint: 'Topology and telemetry' },
  { id: 'intelligence', label: 'Intelligence', hint: 'Agent feed and debate' },
]

function normalizeDebatePayload(payload) {
  if (!payload || typeof payload !== 'object') {
    return null
  }

  if (Array.isArray(payload.arguments) || payload.verdict) {
    return payload
  }

  if (Array.isArray(payload.entries) && payload.final_decision) {
    return {
      action: payload.proposed_action || null,
      arguments: payload.entries.map(entry => ({
        agent: entry.agent_role,
        stance: entry.position,
        confidence: entry.confidence,
        argument: entry.argument,
        conditions: entry.conditions,
      })),
      verdict: {
        decision: String(payload.final_decision).toUpperCase(),
        rationale: payload.judge_rationale,
      },
      consensus_score: payload.consensus_score,
    }
  }

  return null
}

export default function App() {
  const ws = useRef(null)
  const reconnectTimer = useRef(null)
  const isUnmounted = useRef(false)

  const [connected, setConnected] = useState(false)
  const [topology, setTopology] = useState(null)
  const [telemetryHistory, setTelemetryHistory] = useState([])
  const [activeAnomalies, setActiveAnomalies] = useState([])
  const [agentEvents, setAgentEvents] = useState([])
  const [status, setStatus] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [health, setHealth] = useState(null)
  const [debates, setDebates] = useState([])
  const [activeTab, setActiveTab] = useState('operations')

  const apiFetch = useCallback(async (path, opts = {}) => {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    })
    if (!res.ok) {
      throw new Error(`${res.status} ${res.statusText}`)
    }
    return res.json()
  }, [])

  const fetchAll = useCallback(async () => {
    try {
      const [s, m, anomalies] = await Promise.all([
        apiFetch('/api/status'),
        apiFetch('/api/metrics'),
        apiFetch('/api/anomalies'),
      ])
      setStatus(s)
      setMetrics(m)
      setHealth(anomalies?.health || null)
      setActiveAnomalies(Array.isArray(anomalies?.active) ? anomalies.active : [])
    } catch {
      // API may be unavailable during startup.
    }
  }, [apiFetch])

  const fetchTopology = useCallback(async () => {
    try {
      const t = await apiFetch('/api/topology')
      setTopology(t)
    } catch {
      // API may be unavailable during startup.
    }
  }, [apiFetch])

  function connectWs() {
    if (
      isUnmounted.current ||
      ws.current?.readyState === WebSocket.OPEN ||
      ws.current?.readyState === WebSocket.CONNECTING
    ) {
      return
    }

    const socket = new WebSocket(WS_URL)
    ws.current = socket

    socket.onopen = () => {
      setConnected(true)
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
    }

    socket.onmessage = event => {
      let msg = null
      try {
        msg = JSON.parse(event.data)
      } catch {
        return
      }

      const { type, payload, ts } = msg
      const safePayload = payload && typeof payload === 'object' ? payload : {}

      if (type === 'init' && safePayload.snapshot && typeof safePayload.snapshot === 'object') {
        setTelemetryHistory(history => [...history.slice(-(MAX_HISTORY - 1)), safePayload.snapshot])
        return
      }

      if (type === 'telemetry') {
        if (safePayload.snapshot && typeof safePayload.snapshot === 'object') {
          setTelemetryHistory(history => [...history.slice(-(MAX_HISTORY - 1)), safePayload.snapshot])
        }
        if (Array.isArray(safePayload.anomalies)) {
          setActiveAnomalies(safePayload.anomalies)
        }
        if (safePayload.health) {
          setHealth(safePayload.health)
        }
        return
      }

      if (type === 'agent_event') {
        const eventPayload = {
          ...safePayload,
          ts: typeof ts === 'string' ? ts : new Date().toISOString(),
          phase: safePayload.phase || 'observe',
        }
        setAgentEvents(events => [...events.slice(-(MAX_EVENTS - 1)), eventPayload])

        const normalizedDebate = normalizeDebatePayload(safePayload.debate_result)
        if (normalizedDebate) {
          setDebates(current => [
            ...current.slice(-(MAX_DEBATES - 1)),
            { ...normalizedDebate, ts: typeof ts === 'string' ? ts : new Date().toISOString() },
          ])
        }
        return
      }

      if (type === 'kill_switch') {
        setAgentEvents(events => [
          ...events.slice(-(MAX_EVENTS - 1)),
          {
            phase: 'kill_switch',
            message: safePayload.message || 'Kill switch activated.',
            ts: typeof ts === 'string' ? ts : new Date().toISOString(),
          },
        ])
      }
    }

    socket.onclose = () => {
      setConnected(false)
      if (isUnmounted.current) {
        return
      }
      reconnectTimer.current = setTimeout(() => {
        connectWs()
      }, 2500)
    }

    socket.onerror = () => {
      socket.close()
    }
  }

  useEffect(() => {
    isUnmounted.current = false
    const bootFrame = window.requestAnimationFrame(() => {
      connectWs()
      fetchTopology()
      fetchAll()
    })

    const interval = setInterval(fetchAll, 10_000)

    return () => {
      isUnmounted.current = true
      window.cancelAnimationFrame(bootFrame)
      clearInterval(interval)
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
      }
      ws.current?.close()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchTopology, fetchAll])

  const onStart = async () => {
    try {
      await apiFetch('/api/start', { method: 'POST' })
      await fetchAll()
    } catch {
      // no-op
    }
  }

  const onStop = async () => {
    try {
      await apiFetch('/api/stop', { method: 'POST' })
      await fetchAll()
    } catch {
      // no-op
    }
  }

  const onInject = async payload => {
    try {
      await apiFetch('/api/inject', {
        method: 'POST',
        body: JSON.stringify(payload),
      })
    } catch {
      // no-op
    }
  }

  const onKillSwitch = async () => {
    try {
      await apiFetch('/api/kill-switch', { method: 'POST' })
      await fetchAll()
    } catch {
      // no-op
    }
  }

  const healthLabel = (health?.overall_health || 'unknown').toUpperCase()
  const latestTelemetry = telemetryHistory.length > 0 ? telemetryHistory[telemetryHistory.length - 1] : null

  return (
    <div className="relative min-h-screen overflow-x-hidden pb-4">
      <header className="sticky top-0 z-20 border-b border-[rgba(226,243,16,0.12)] bg-[rgba(4,7,4,0.86)] px-4 py-3 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1520px] items-center gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="ui-pill tracking-[0.18em] uppercase">
                B102
              </span>
            </div>
            <span className="mt-1 block text-sm font-semibold tracking-[0.08em] text-[var(--headline)]">
              Ballmer Agentic Conception - BAC
            </span>
            <span className="block truncate text-[11px] text-[var(--muted)]">
              Cyber Chakravyuh
            </span>
          </div>

          <div className="ml-auto flex items-center gap-2 text-xs">
            <span className={`flex items-center gap-1.5 ${connected ? 'text-[var(--ok)]' : 'text-[var(--critical)]'}`}>
              <span className={`h-2 w-2 rounded-full ${connected ? 'bg-[var(--ok)]' : 'bg-[var(--critical)]'}`} />
              {connected ? 'Live stream' : 'Disconnected'}
            </span>
            <span className="ui-pill mono">
              Cycle {status?.orchestrator?.cycle_count ?? '—'}
            </span>
            <span className={`ui-pill ${
              health?.overall_health === 'critical'
                ? 'border-[rgba(255,93,108,0.48)] text-[var(--critical)]'
                : health?.overall_health === 'degraded'
                  ? 'border-[rgba(255,159,67,0.45)] text-[var(--high)]'
                  : 'border-[rgba(83,216,106,0.42)] text-[var(--ok)]'
            }`}>
              {healthLabel}
            </span>
            {status?.kill_switch_active && (
              <span className="ui-pill border-[rgba(255,93,108,0.52)] bg-[rgba(255,93,108,0.1)] font-semibold text-[var(--critical)]">
                Kill Switch
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1320px] px-4 pt-4">
        <div className="ui-tabs mb-4">
          {TABS.map(tab => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`ui-tab ${activeTab === tab.id ? 'ui-tab-active' : ''}`}
              aria-pressed={activeTab === tab.id}
            >
              <span>{tab.label}</span>
              <span className="ui-tab-hint">{tab.hint}</span>
            </button>
          ))}
        </div>

        {activeTab === 'operations' && (
          <section className="grid grid-cols-1 gap-4 xl:grid-cols-12">
            <div className="xl:col-span-4">
              <MagicCard>
                <ControlPanel
                  status={status}
                  onStart={onStart}
                  onStop={onStop}
                  onInject={onInject}
                  onKillSwitch={onKillSwitch}
                />
              </MagicCard>
            </div>

            <div className="xl:col-span-4">
              <MagicCard>
                <MetricsPanel metrics={metrics} health={health} />
              </MagicCard>
            </div>

            <div className="xl:col-span-4">
              <MagicCard>
                <h2 className="ui-title mb-2">
                  Active Anomalies
                </h2>
                {activeAnomalies.length === 0 ? (
                  <div className="ui-empty py-4">
                    No active anomalies
                  </div>
                ) : (
                  <div className="max-h-[460px] space-y-1 overflow-y-auto">
                    {activeAnomalies.map((anomaly, index) => {
                      const entity = anomaly.node_id || anomaly.link_id || 'unknown'
                      const numericValue = Number(anomaly.observed_value)
                      const value = Number.isFinite(numericValue) ? numericValue.toFixed(2) : '—'
                      return (
                        <div
                          key={anomaly.id || `${entity}_${anomaly.metric_name}_${index}`}
                          className="ui-row flex items-center gap-2 py-1.5 text-xs"
                        >
                          <span className={`badge-${anomaly.severity}`}>{anomaly.severity}</span>
                          <span className="min-w-0 flex-1 truncate text-[var(--muted)]">
                            {entity} · {anomaly.metric_name}
                          </span>
                          <span className="mono text-[var(--headline)]">{value}</span>
                        </div>
                      )
                    })}
                  </div>
                )}
              </MagicCard>
            </div>
          </section>
        )}

        {activeTab === 'network' && (
          <section className="space-y-4">
            <MagicCard className="h-[390px] md:h-[520px]">
              <TopologyGraph topology={topology} anomalies={activeAnomalies} telemetry={latestTelemetry} />
            </MagicCard>

            <MagicCard>
              <TelemetryCharts history={telemetryHistory} />
            </MagicCard>
          </section>
        )}

        {activeTab === 'intelligence' && (
          <section className="grid grid-cols-1 items-start gap-4 xl:grid-cols-12">
            <MagicCard className="xl:col-span-4">
              <AgentFeed events={agentEvents} />
            </MagicCard>

            <MagicCard className="xl:col-span-8">
              <DebateViewer debates={debates} />
            </MagicCard>
          </section>
        )}
      </main>
    </div>
  )
}
