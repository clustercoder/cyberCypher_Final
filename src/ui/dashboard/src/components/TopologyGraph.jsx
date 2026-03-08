import { useEffect, useMemo, useRef } from 'react'
import * as d3 from 'd3'

const NODE_COLOR = {
  core: '#7fa4c8',
  aggregation: '#3D5E5D',
  edge: '#47B5B8',
  peering: '#d2a17a',
}

const SEVERITY_COLOR = {
  critical: '#d67f89',
  high: '#d2a17a',
  medium: '#c8b27b',
  low: '#7fa4c8',
}

function severityRank(level) {
  return { critical: 4, high: 3, medium: 2, low: 1 }[level] || 0
}

function buildAnomalyMap(anomalies) {
  const map = {}
  for (const anomaly of anomalies || []) {
    const key = anomaly.node_id || anomaly.link_id
    if (!key) continue
    if (!map[key] || severityRank(anomaly.severity) > severityRank(map[key].severity)) {
      map[key] = anomaly
    }
  }
  return map
}

function linkAnomaly(anomalyMap, source, target) {
  return (
    anomalyMap[`${source}-${target}`] ||
    anomalyMap[`${target}-${source}`] ||
    anomalyMap[[source, target].sort().join('-')]
  )
}

function readLinkMetrics(telemetry, source, target) {
  const links = telemetry?.links
  if (!links || typeof links !== 'object') return null

  const direct = links[`${source}-${target}`]
  if (direct && typeof direct === 'object') return direct

  const reverse = links[`${target}-${source}`]
  if (reverse && typeof reverse === 'object') return reverse

  const sorted = links[[source, target].sort().join('-')]
  if (sorted && typeof sorted === 'object') return sorted

  return null
}

function readNodeMetrics(telemetry, nodeId) {
  const nodes = telemetry?.nodes
  if (!nodes || typeof nodes !== 'object') return null
  const metrics = nodes[nodeId]
  return metrics && typeof metrics === 'object' ? metrics : null
}

function toNumber(value) {
  const num = Number(value)
  return Number.isFinite(num) ? num : null
}

function fmt(value, digits = 1, suffix = '') {
  const num = toNumber(value)
  if (num == null) return '—'
  return `${num.toFixed(digits)}${suffix}`
}

function linkBaseStyle(utilizationPct) {
  if (utilizationPct == null) {
    return {
      color: 'rgba(61,94,93,0.28)',
      width: 1.6,
      opacity: 0.5,
    }
  }

  if (utilizationPct >= 85) {
    return {
      color: '#c8a190',
      width: 2.4,
      opacity: 0.85,
    }
  }

  if (utilizationPct >= 70) {
    return {
      color: '#47B5B8',
      width: 2.1,
      opacity: 0.75,
    }
  }

  if (utilizationPct >= 50) {
    return {
      color: 'rgba(71,181,184,0.5)',
      width: 1.9,
      opacity: 0.65,
    }
  }

  return {
    color: 'rgba(61,94,93,0.34)',
    width: 1.7,
    opacity: 0.55,
  }
}

export default function TopologyGraph({ topology, anomalies = [], telemetry = null }) {
  const svgRef = useRef(null)
  const anomalyMap = useMemo(() => buildAnomalyMap(anomalies), [anomalies])
  const anomalyMapRef = useRef(anomalyMap)
  const telemetryRef = useRef(telemetry)

  useEffect(() => {
    anomalyMapRef.current = anomalyMap
  }, [anomalyMap])

  useEffect(() => {
    telemetryRef.current = telemetry
  }, [telemetry])

  useEffect(() => {
    if (!topology || !svgRef.current) return

    const svgNode = svgRef.current
    const width = svgNode.clientWidth || 700
    const height = svgNode.clientHeight || 420

    const svg = d3.select(svgNode)
    svg.selectAll('*').remove()

    const defs = svg.append('defs')
    Object.entries(SEVERITY_COLOR).forEach(([severity, color]) => {
      const filter = defs
        .append('filter')
        .attr('id', `topo-glow-${severity}`)
        .attr('x', '-35%')
        .attr('y', '-35%')
        .attr('width', '170%')
        .attr('height', '170%')

      filter
        .append('feDropShadow')
        .attr('dx', 0)
        .attr('dy', 0)
        .attr('stdDeviation', 2)
        .attr('flood-color', color)
        .attr('flood-opacity', 0.28)
    })

    const g = svg.append('g')
    const zoom = d3.zoom().scaleExtent([0.45, 2.6]).on('zoom', event => g.attr('transform', event.transform))
    svg.call(zoom)

    const nodes = (topology.nodes || []).map(node => ({ ...node }))
    const links = (topology.links || []).map(link => ({ ...link }))

    const simulation = d3
      .forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(node => node.id).distance(98).strength(0.75))
      .force('charge', d3.forceManyBody().strength(-260))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(30))

    const linkLayer = g.append('g').attr('class', 'links')
    const pulseLayer = g.append('g').attr('class', 'pulses')
    const nodeLayer = g.append('g').attr('class', 'nodes')
    const metricLayer = g.append('g').attr('class', 'link-metrics')

    const linkSelection = linkLayer
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke-linecap', 'round')

    const nodeSelection = nodeLayer
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', 14)
      .attr('stroke-width', 1.2)
      .style('cursor', 'grab')
      .call(
        d3
          .drag()
          .on('start', (event, node) => {
            if (!event.active) simulation.alphaTarget(0.25).restart()
            node.fx = node.x
            node.fy = node.y
          })
          .on('drag', (event, node) => {
            node.fx = event.x
            node.fy = event.y
          })
          .on('end', (event, node) => {
            if (!event.active) simulation.alphaTarget(0)
            node.fx = null
            node.fy = null
          }),
      )

    const labels = g
      .append('g')
      .selectAll('text')
      .data(nodes)
      .join('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.34em')
      .attr('font-size', 7)
      .attr('font-weight', 700)
      .attr('fill', '#f7f9f0')
      .attr('pointer-events', 'none')
      .text(node => node.id)

    const linkMetricLabels = metricLayer
      .selectAll('text')
      .data(links)
      .join('text')
      .attr('font-size', 7)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(71,181,184,0.65)')
      .attr('pointer-events', 'none')

    const tooltip = d3
      .select('body')
      .selectAll('.topology-tooltip')
      .data([null])
      .join('div')
      .attr('class', 'topology-tooltip')
      .style('position', 'fixed')
      .style('pointer-events', 'none')
      .style('display', 'none')
      .style('z-index', 9999)
      .style('max-width', '260px')
      .style('border-radius', '10px')
      .style('border', '1px solid rgba(71,181,184,0.28)')
      .style('background', 'rgba(8,10,8,0.96)')
      .style('padding', '8px 10px')
      .style('font-size', '11px')
      .style('box-shadow', '0 12px 20px rgba(0,0,0,0.28)')
      .style('color', '#d7dfcd')

    function repaintFromState() {
      const map = anomalyMapRef.current
      const currentTelemetry = telemetryRef.current

      linkSelection
        .attr('stroke', link => {
          const source = link.source?.id || link.source
          const target = link.target?.id || link.target
          const anomaly = linkAnomaly(map, source, target)
          const util = toNumber(readLinkMetrics(currentTelemetry, source, target)?.utilization_pct)
          const base = linkBaseStyle(util)
          return anomaly ? SEVERITY_COLOR[anomaly.severity] : base.color
        })
        .attr('stroke-width', link => {
          const source = link.source?.id || link.source
          const target = link.target?.id || link.target
          const anomaly = linkAnomaly(map, source, target)
          const util = toNumber(readLinkMetrics(currentTelemetry, source, target)?.utilization_pct)
          const base = linkBaseStyle(util)
          return anomaly ? Math.max(2.8, base.width) : base.width
        })
        .attr('opacity', link => {
          const source = link.source?.id || link.source
          const target = link.target?.id || link.target
          const anomaly = linkAnomaly(map, source, target)
          const util = toNumber(readLinkMetrics(currentTelemetry, source, target)?.utilization_pct)
          const base = linkBaseStyle(util)
          return anomaly ? 0.94 : base.opacity
        })

      nodeSelection
        .attr('fill', node => {
          const anomaly = map[node.id]
          return anomaly ? SEVERITY_COLOR[anomaly.severity] : NODE_COLOR[node.node_type] || '#98a68d'
        })
        .attr('stroke', node => (map[node.id] ? '#f7f9f0' : 'rgba(71,181,184,0.34)'))
        .attr('filter', node => {
          const anomaly = map[node.id]
          return anomaly ? `url(#topo-glow-${anomaly.severity})` : null
        })

      pulseLayer.selectAll('*').remove()
      const anomalousNodes = nodeSelection.data().filter(node => map[node.id])
      pulseLayer
        .selectAll('circle')
        .data(anomalousNodes)
        .join('circle')
        .attr('class', 'pulse-ring')
        .attr('r', 14)
        .attr('fill', 'none')
        .attr('stroke', node => SEVERITY_COLOR[map[node.id].severity])
        .attr('stroke-width', 1.1)
        .attr('opacity', 0.55)

      linkMetricLabels
        .text(link => {
          const source = link.source?.id || link.source
          const target = link.target?.id || link.target
          const util = toNumber(readLinkMetrics(currentTelemetry, source, target)?.utilization_pct)
          const anomaly = linkAnomaly(map, source, target)
          if (util == null) return ''
          if (!anomaly && util < 70) return ''
          return `${Math.round(util)}%`
        })
        .attr('fill', link => {
          const source = link.source?.id || link.source
          const target = link.target?.id || link.target
          const anomaly = linkAnomaly(map, source, target)
          if (anomaly) return SEVERITY_COLOR[anomaly.severity]
          return 'rgba(71,181,184,0.65)'
        })
        .attr('opacity', link => {
          const source = link.source?.id || link.source
          const target = link.target?.id || link.target
          const util = toNumber(readLinkMetrics(currentTelemetry, source, target)?.utilization_pct)
          const anomaly = linkAnomaly(map, source, target)
          if (anomaly) return 0.9
          return util != null && util >= 70 ? 0.75 : 0
        })
    }

    nodeSelection
      .on('mouseover', (event, node) => {
        const anomaly = anomalyMapRef.current[node.id]
        const metrics = readNodeMetrics(telemetryRef.current, node.id)
        const anomalyLine = anomaly
          ? `<div style="margin-top:4px;color:${SEVERITY_COLOR[anomaly.severity]};font-weight:600;">${anomaly.severity.toUpperCase()} · ${anomaly.metric_name}: ${fmt(anomaly.observed_value, 2)}</div>`
          : ''

        tooltip
          .style('display', 'block')
          .html(
            `<div style="font-weight:700;color:#f7f9f0;">${node.id}</div>
             <div style="color:#98a68d;">Type: ${node.node_type || 'unknown'}</div>
             <div style="margin-top:4px;color:#d7dfcd;">CPU: ${fmt(metrics?.cpu_pct, 1, '%')} · Memory: ${fmt(metrics?.memory_pct, 1, '%')}</div>
             <div style="color:#d7dfcd;">Temp: ${fmt(metrics?.temperature_c, 1, '°C')}</div>${anomalyLine}`,
          )
      })
      .on('mousemove', event => {
        tooltip
          .style('left', `${event.clientX + 14}px`)
          .style('top', `${event.clientY - 8}px`)
      })
      .on('mouseout', () => {
        tooltip.style('display', 'none')
      })

    linkSelection
      .on('mouseover', (event, link) => {
        const source = link.source?.id || link.source
        const target = link.target?.id || link.target
        const metrics = readLinkMetrics(telemetryRef.current, source, target)
        const anomaly = linkAnomaly(anomalyMapRef.current, source, target)
        const anomalyLine = anomaly
          ? `<div style="margin-top:4px;color:${SEVERITY_COLOR[anomaly.severity]};font-weight:600;">${anomaly.severity.toUpperCase()} · ${anomaly.metric_name}: ${fmt(anomaly.observed_value, 2)}</div>`
          : ''

        tooltip
          .style('display', 'block')
          .html(
            `<div style="font-weight:700;color:#f7f9f0;">${source}-${target}</div>
             <div style="margin-top:4px;color:#d7dfcd;">Utilization: ${fmt(metrics?.utilization_pct, 1, '%')}</div>
             <div style="color:#d7dfcd;">Latency: ${fmt(metrics?.latency_ms, 2, 'ms')} · Loss: ${fmt(metrics?.packet_loss_pct, 2, '%')}</div>
             <div style="color:#d7dfcd;">Throughput: ${fmt(metrics?.throughput_gbps, 2, 'Gbps')}</div>${anomalyLine}`,
          )
      })
      .on('mousemove', event => {
        tooltip
          .style('left', `${event.clientX + 14}px`)
          .style('top', `${event.clientY - 8}px`)
      })
      .on('mouseout', () => {
        tooltip.style('display', 'none')
      })

    simulation.on('tick', () => {
      linkSelection
        .attr('x1', link => link.source.x)
        .attr('y1', link => link.source.y)
        .attr('x2', link => link.target.x)
        .attr('y2', link => link.target.y)

      nodeSelection
        .attr('cx', node => node.x)
        .attr('cy', node => node.y)

      labels
        .attr('x', node => node.x)
        .attr('y', node => node.y)

      pulseLayer
        .selectAll('.pulse-ring')
        .attr('cx', node => node.x)
        .attr('cy', node => node.y)

      linkMetricLabels
        .attr('x', link => ((link.source.x || 0) + (link.target.x || 0)) / 2)
        .attr('y', link => ((link.source.y || 0) + (link.target.y || 0)) / 2 - 4)
    })

    repaintFromState()

    return () => {
      simulation.stop()
      tooltip.remove()
    }
  }, [topology])

  useEffect(() => {
    if (!svgRef.current) return
    const svg = d3.select(svgRef.current)
    const map = anomalyMap
    const currentTelemetry = telemetry

    svg.selectAll('.nodes circle')
      .attr('fill', node => {
        const anomaly = map[node.id]
        return anomaly ? SEVERITY_COLOR[anomaly.severity] : NODE_COLOR[node.node_type] || '#98a68d'
      })
      .attr('stroke', node => (map[node.id] ? '#f7f9f0' : 'rgba(71,181,184,0.34)'))
      .attr('filter', node => {
        const anomaly = map[node.id]
        return anomaly ? `url(#topo-glow-${anomaly.severity})` : null
      })

    svg.selectAll('.links line')
      .attr('stroke', link => {
        const source = link.source?.id || link.source
        const target = link.target?.id || link.target
        const anomaly = linkAnomaly(map, source, target)
        const util = toNumber(readLinkMetrics(currentTelemetry, source, target)?.utilization_pct)
        const base = linkBaseStyle(util)
        return anomaly ? SEVERITY_COLOR[anomaly.severity] : base.color
      })
      .attr('stroke-width', link => {
        const source = link.source?.id || link.source
        const target = link.target?.id || link.target
        const anomaly = linkAnomaly(map, source, target)
        const util = toNumber(readLinkMetrics(currentTelemetry, source, target)?.utilization_pct)
        const base = linkBaseStyle(util)
        return anomaly ? Math.max(2.8, base.width) : base.width
      })
      .attr('opacity', link => {
        const source = link.source?.id || link.source
        const target = link.target?.id || link.target
        const anomaly = linkAnomaly(map, source, target)
        const util = toNumber(readLinkMetrics(currentTelemetry, source, target)?.utilization_pct)
        const base = linkBaseStyle(util)
        return anomaly ? 0.94 : base.opacity
      })

    const nodeData = svg.selectAll('.nodes circle').data()
    const pulseLayer = svg.select('.pulses')
    pulseLayer.selectAll('*').remove()
    pulseLayer
      .selectAll('circle')
      .data(nodeData.filter(node => map[node.id]))
      .join('circle')
      .attr('class', 'pulse-ring')
      .attr('r', 14)
      .attr('fill', 'none')
      .attr('stroke', node => SEVERITY_COLOR[map[node.id].severity])
      .attr('stroke-width', 1.1)
      .attr('opacity', 0.55)
      .attr('cx', node => node.x)
      .attr('cy', node => node.y)

    svg.selectAll('.link-metrics text')
      .text(link => {
        const source = link.source?.id || link.source
        const target = link.target?.id || link.target
        const util = toNumber(readLinkMetrics(currentTelemetry, source, target)?.utilization_pct)
        const anomaly = linkAnomaly(map, source, target)
        if (util == null) return ''
        if (!anomaly && util < 70) return ''
        return `${Math.round(util)}%`
      })
      .attr('fill', link => {
        const source = link.source?.id || link.source
        const target = link.target?.id || link.target
        const anomaly = linkAnomaly(map, source, target)
        return anomaly ? SEVERITY_COLOR[anomaly.severity] : 'rgba(71,181,184,0.65)'
      })
      .attr('opacity', link => {
        const source = link.source?.id || link.source
        const target = link.target?.id || link.target
        const util = toNumber(readLinkMetrics(currentTelemetry, source, target)?.utilization_pct)
        const anomaly = linkAnomaly(map, source, target)
        if (anomaly) return 0.9
        return util != null && util >= 70 ? 0.75 : 0
      })
  }, [anomalyMap, telemetry])

  return (
    <div className="flex h-full flex-col">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="ui-title">Network Topology</h2>
        <div className="flex flex-wrap gap-3">
          {[
            { label: 'Core', color: NODE_COLOR.core },
            { label: 'Aggregation', color: NODE_COLOR.aggregation },
            { label: 'Edge', color: NODE_COLOR.edge },
            { label: 'Peering', color: NODE_COLOR.peering },
            { label: 'Anomaly', color: SEVERITY_COLOR.critical },
          ].map(item => (
            <div key={item.label} className="flex items-center gap-1.5 mono text-[10px] text-[var(--muted)]">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: item.color }} />
              {item.label}
            </div>
          ))}
          <span className="mono text-[10px] text-[var(--muted)]">hover for live metrics</span>
        </div>
      </div>
      <svg
        ref={svgRef}
        className="h-full w-full rounded-xl border border-[rgba(71,181,184,0.14)] bg-[rgba(255,255,255,0.01)]"
      />
    </div>
  )
}
