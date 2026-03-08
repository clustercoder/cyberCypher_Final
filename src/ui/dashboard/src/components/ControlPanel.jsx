import { useMemo, useState } from 'react'
import ShimmerButton from './magicui/ShimmerButton'

const SCENARIOS = [
  { value: 'congestion_cascade', label: 'Congestion cascade' },
  { value: 'ddos_surge', label: 'DDoS surge' },
  { value: 'fiber_cut', label: 'Fiber cut' },
  { value: 'misconfiguration', label: 'Misconfiguration' },
  { value: 'hardware_degradation', label: 'Hardware degradation' },
]

const SUGGESTED_TARGETS = ['CR1-CR2', 'CR1-AGG1', 'CR1-AGG2', 'CR2-AGG3', 'CR2-AGG4']

export default function ControlPanel({ status, onStart, onStop, onInject, onKillSwitch }) {
  const [scenario, setScenario] = useState('congestion_cascade')
  const [target, setTarget] = useState('CR1-CR2')
  const [duration, setDuration] = useState(10)
  const [injecting, setInjecting] = useState(false)

  const isRunning = status?.status === 'running' && status?.agent_running
  const killActive = Boolean(status?.kill_switch_active)

  const statusLabel = useMemo(() => {
    if (killActive) return 'Kill switch engaged'
    if (isRunning) return 'Agent running'
    return 'Agent stopped'
  }, [isRunning, killActive])

  const handleInject = async () => {
    setInjecting(true)
    try {
      await onInject({ scenario_type: scenario, target, duration_minutes: duration })
    } finally {
      setInjecting(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="ui-title">
          Control Panel
        </h2>
        <span className="ui-pill mono">
          Ticks {status?.engine_ticks ?? '—'}
        </span>
      </div>

      <div className="ui-panel-block">
        <div className="mb-2 flex items-center gap-2 text-xs">
          <span className={`h-2 w-2 rounded-full ${isRunning ? 'bg-[var(--ok)]' : 'bg-[var(--muted)]'}`} />
          <span className="text-[var(--muted)]">{statusLabel}</span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <ShimmerButton
            onClick={onStart}
            disabled={isRunning || killActive}
            variant="primary"
          >
            Start Agent
          </ShimmerButton>
          <ShimmerButton
            onClick={onStop}
            disabled={!isRunning}
            variant="secondary"
          >
            Stop Agent
          </ShimmerButton>
        </div>
      </div>

      <div className="ui-panel-block space-y-2">
        <h3 className="ui-subtitle">
          Inject anomaly
        </h3>
        <select
          value={scenario}
          onChange={event => setScenario(event.target.value)}
          className="ui-field"
        >
          {SCENARIOS.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>

        <div className="grid grid-cols-[1fr_auto] gap-2">
          <input
            value={target}
            onChange={event => setTarget(event.target.value)}
            placeholder="Target (CR1-CR2)"
            className="ui-field"
          />
          <input
            type="number"
            min={1}
            max={120}
            value={duration}
            onChange={event => setDuration(Number(event.target.value) || 10)}
            className="ui-field w-20"
            aria-label="Duration minutes"
          />
        </div>

        <div className="flex flex-wrap gap-1.5">
          {SUGGESTED_TARGETS.map(option => (
            <button
              key={option}
              type="button"
              onClick={() => setTarget(option)}
              className={`ui-chip transition ${
                target === option
                  ? 'ui-chip-active'
                  : ''
              }`}
            >
              {option}
            </button>
          ))}
        </div>

        <ShimmerButton
          onClick={handleInject}
          disabled={injecting || killActive}
          variant="primary"
          className="w-full"
        >
          {injecting ? 'Injecting…' : 'Inject Scenario'}
        </ShimmerButton>
      </div>

      <div className="ui-panel-block border-[rgba(255,93,108,0.34)] bg-[rgba(255,93,108,0.06)]">
        <ShimmerButton
          onClick={() => {
            if (window.confirm('Activate kill switch? This halts all autonomous actions.')) {
              onKillSwitch()
            }
          }}
          disabled={killActive}
          variant="danger"
          className="w-full"
        >
          {killActive ? 'Kill Switch Active' : 'Activate Kill Switch'}
        </ShimmerButton>
      </div>
    </div>
  )
}
