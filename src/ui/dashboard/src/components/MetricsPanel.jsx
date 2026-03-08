function formatNumber(value, decimals = 1) {
  if (value == null || Number.isNaN(value)) return '—'
  return Number(value).toFixed(decimals)
}

function Stat({ label, value, unit = '', subtitle, color = 'text-[var(--headline)]' }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="mono text-[10px] uppercase tracking-[0.12em] text-[var(--muted)]">{label}</span>
      <span key={`${label}-${value}`} className={`value-flash mono text-xl font-semibold ${color}`}>
        {value != null ? `${value}${unit}` : '—'}
      </span>
      {subtitle ? <span className="text-[10px] text-[var(--muted)]">{subtitle}</span> : null}
    </div>
  )
}

function HealthBar({ health }) {
  if (!health) return null
  const total = Math.max(health.total_anomalies || 0, 1)
  const bySeverity = health.by_severity || {}
  const criticalPct = Math.min(((bySeverity.critical || 0) / total) * 100, 100)
  const highPct = Math.min(((bySeverity.high || 0) / total) * 100, 100)
  const mediumPct = Math.min(((bySeverity.medium || 0) / total) * 100, 100)

  return (
    <div className="mt-2 flex h-2 overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
      <div className="bg-[var(--critical)] transition-all duration-500" style={{ width: `${criticalPct}%` }} />
      <div className="bg-[var(--high)] transition-all duration-500" style={{ width: `${highPct}%` }} />
      <div className="bg-[var(--medium)] transition-all duration-500" style={{ width: `${mediumPct}%` }} />
      <div className="flex-1 bg-[rgba(71,181,184,0.24)]" />
    </div>
  )
}

export default function MetricsPanel({ metrics, health }) {
  const healthClass =
    health?.overall_health === 'critical'
      ? 'health-critical'
      : health?.overall_health === 'degraded'
        ? 'health-degraded'
        : health?.overall_health === 'healthy'
          ? 'health-healthy'
          : 'text-[var(--muted)]'

  return (
    <div className={`space-y-4`}>
      <h2 className="ui-title">
        System Metrics
      </h2>

      <div>
        <div className="ui-subtitle">Network Health</div>
        <div className={`mono text-lg font-semibold tracking-wide ${healthClass}`}>
          {health?.overall_health?.toUpperCase() || '—'}
        </div>
        <HealthBar health={health} />
        {health ? (
          <div className="mt-1 text-[10px] text-[var(--muted)]">
            {health.total_anomalies} anomalies active
            {health.worst_entity ? ` · ${health.worst_entity}` : ''}
          </div>
        ) : null}
      </div>

      <div className="ui-panel-block grid grid-cols-2 gap-3">
        <Stat
          label="MTTD"
          value={metrics?.mttd_seconds != null ? formatNumber(metrics.mttd_seconds, 1) : null}
          unit="s"
          subtitle="Mean detection time"
          color="text-[var(--accent)]"
        />
        <Stat
          label="MTTM"
          value={metrics?.mttm_seconds != null ? formatNumber(metrics.mttm_seconds, 1) : null}
          unit="s"
          subtitle="Mean mitigation time"
          color="text-[var(--ok)]"
        />
      </div>

      <div className="ui-panel-block grid grid-cols-3 gap-3">
        <Stat
          label="Precision"
          value={metrics?.precision != null ? formatNumber(metrics.precision * 100, 1) : null}
          unit="%"
          color="text-[var(--high)]"
        />
        <Stat
          label="Recall"
          value={metrics?.recall != null ? formatNumber(metrics.recall * 100, 1) : null}
          unit="%"
          color="text-[var(--accent-2)]"
        />
        <Stat
          label="F1"
          value={metrics?.f1 != null ? formatNumber(metrics.f1 * 100, 1) : null}
          unit="%"
          color="text-[var(--headline)]"
        />
      </div>

      <div className="ui-panel-block grid grid-cols-2 gap-3">
        <Stat label="Anomalies" value={metrics?.anomalies_detected ?? null} color="text-[var(--high)]" />
        <Stat label="Actions" value={metrics?.actions_executed ?? null} color="text-[var(--headline)]" />
        <Stat label="Successful" value={metrics?.actions_successful ?? null} color="text-[var(--ok)]" />
        <Stat label="Failed" value={metrics?.actions_failed ?? null} color="text-[var(--critical)]" />
      </div>
    </div>
  )
}
