import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const CHART_COLORS = ['#e2f310', '#54f5c2', '#67b7ff', '#ff9f43', '#e182ff']

function formatTimeLabel(timestamp) {
  if (!timestamp) return '--:--:--'
  const date = new Date(timestamp)
  if (Number.isNaN(date.getTime())) return '--:--:--'
  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function buildMetricSeries(history, entityKeys, metric) {
  return history.map(snapshot => {
    const ts = formatTimeLabel(snapshot?.timestamp)
    const row = { ts }
    for (const key of entityKeys) {
      const links = snapshot?.links && typeof snapshot.links === 'object' ? snapshot.links : {}
      const nodes = snapshot?.nodes && typeof snapshot.nodes === 'object' ? snapshot.nodes : {}
      const pool = links[key] ? links : nodes
      row[key] = pool?.[key]?.[metric] ?? null
    }
    return row
  })
}

function pickEntityKeys(history) {
  const latest = history[history.length - 1] || {}
  const links = Object.keys(latest?.links && typeof latest.links === 'object' ? latest.links : {})
  const nodes = Object.keys(latest?.nodes && typeof latest.nodes === 'object' ? latest.nodes : {})
  return {
    links: links.slice(0, 3),
    nodes: nodes.slice(0, 3),
  }
}

function MetricChart({ title, data, keys, threshold, domain }) {
  return (
    <div className="ui-panel-block">
      <h3 className="ui-subtitle mb-2">
        {title}
      </h3>
      <ResponsiveContainer width="100%" height={190}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -24, bottom: 0 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="rgba(226,243,16,0.12)" />
          <XAxis dataKey="ts" tick={{ fontSize: 10, fill: '#96a38a' }} tickLine={false} />
          <YAxis
            domain={domain || ['auto', 'auto']}
            tick={{ fontSize: 10, fill: '#96a38a' }}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              background: 'rgba(10,14,10,0.95)',
              border: '1px solid rgba(226,243,16,0.28)',
              borderRadius: '0.65rem',
              fontSize: 11,
              color: '#dce4d2',
            }}
            labelStyle={{ color: '#f7f9f0' }}
          />
          <Legend wrapperStyle={{ fontSize: 10, color: '#96a38a' }} />
          {threshold != null ? (
            <ReferenceLine y={threshold} stroke="#ff5d6c" strokeDasharray="4 3" />
          ) : null}
          {keys.map((key, index) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={CHART_COLORS[index % CHART_COLORS.length]}
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function TelemetryCharts({ history }) {
  if (!history?.length) {
    return (
      <div className="ui-empty flex h-32 items-center justify-center text-sm">
        Waiting for telemetry…
      </div>
    )
  }

  const { links: linkKeys, nodes: nodeKeys } = pickEntityKeys(history)
  if (!linkKeys.length && !nodeKeys.length) {
    return (
      <div className="ui-empty flex h-32 items-center justify-center text-sm">
        Telemetry stream connected, awaiting entities…
      </div>
    )
  }

  const utilizationData = buildMetricSeries(history, linkKeys, 'utilization_pct')
  const latencyData = buildMetricSeries(history, linkKeys, 'latency_ms')
  const cpuData = buildMetricSeries(history, nodeKeys, 'cpu_pct')
  const packetLossData = buildMetricSeries(history, linkKeys, 'packet_loss_pct')

  return (
    <div className="space-y-3">
      {linkKeys.length ? (
        <MetricChart title="Link Utilization (%)" data={utilizationData} keys={linkKeys} threshold={85} domain={[0, 100]} />
      ) : null}
      {linkKeys.length ? (
        <MetricChart title="Link Latency (ms)" data={latencyData} keys={linkKeys} />
      ) : null}
      {nodeKeys.length ? (
        <MetricChart title="Node CPU (%)" data={cpuData} keys={nodeKeys} threshold={85} domain={[0, 100]} />
      ) : null}
      {linkKeys.length ? (
        <MetricChart title="Packet Loss (%)" data={packetLossData} keys={linkKeys} threshold={1} />
      ) : null}
    </div>
  )
}
