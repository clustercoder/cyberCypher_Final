const AGENT_COLOR = {
  reliability: 'text-[#84caff]',
  performance: 'text-[var(--ok)]',
  cost_sla: 'text-[var(--high)]',
  judge: 'text-[#e182ff]',
  ReliabilityAgent: 'text-[#84caff]',
  PerformanceAgent: 'text-[var(--ok)]',
  CostSLAAgent: 'text-[var(--high)]',
  JudgeAgent: 'text-[#e182ff]',
}

function normalizeEntry(entry) {
  const role = entry?.agent_role || entry?.agent || 'specialist'
  return {
    role,
    position: (entry?.position || entry?.stance || 'conditional').toLowerCase(),
    argument: entry?.argument || 'No argument provided.',
    confidence: typeof entry?.confidence === 'number' ? entry.confidence : null,
    conditions: Array.isArray(entry?.conditions) ? entry.conditions : [],
  }
}

function positionColor(position) {
  if (position === 'support' || position === 'approve') return 'text-[var(--ok)]'
  if (position === 'oppose' || position === 'reject') return 'text-[var(--critical)]'
  return 'text-[var(--muted)]'
}

function DebateArg({ entry }) {
  const color = AGENT_COLOR[entry.role] || 'text-[var(--text)]'
  const sentimentColor = positionColor(entry.position)

  return (
    <div className="ui-panel-block space-y-1 p-2.5">
      <div className="flex items-center gap-2">
        <span className={`text-[11px] font-semibold uppercase tracking-[0.12em] ${color}`}>{entry.role}</span>
        <span className={`text-[10px] uppercase tracking-[0.1em] ${sentimentColor}`}>
          {entry.position}
        </span>
        {entry.confidence != null ? (
          <span className="mono ml-auto text-[10px] text-[var(--muted)]">
            conf {entry.confidence.toFixed(2)}
          </span>
        ) : null}
      </div>
      <p className="text-xs leading-relaxed text-[var(--text)]">{entry.argument}</p>
      {entry.conditions.length > 0 ? (
        <p className="text-[10px] text-[var(--muted)]">
          Conditions: {entry.conditions.join(', ')}
        </p>
      ) : null}
    </div>
  )
}

export default function DebateViewer({ debates = [] }) {
  if (!debates.length) {
    return (
      <div className="space-y-2">
        <h2 className="ui-title mb-2">
          Debate Viewer
        </h2>
        <div className="ui-empty py-10">
          No debates conducted yet
        </div>
      </div>
    )
  }

  const latest = debates[debates.length - 1]
  const entries = (latest.arguments || latest.entries || []).map(normalizeEntry)
  const verdict = latest.verdict || {
    decision: latest.final_decision ? String(latest.final_decision).toUpperCase() : 'MODIFY',
    rationale: latest.judge_rationale || latest.rationale || 'No rationale available.',
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="ui-title">
          Debate Viewer
        </h2>
        <span className="ui-pill">{debates.length} debate(s)</span>
      </div>

      <div className="ui-panel-block p-2.5">
        <div className="mb-1 flex items-center gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-[0.1em] text-[#e182ff]">Judge Verdict</span>
          <span className={`text-[11px] font-semibold ${positionColor(String(verdict.decision || '').toLowerCase())}`}>
            {verdict.decision}
          </span>
          {typeof latest.consensus_score === 'number' ? (
            <span className="mono ml-auto text-[10px] text-[var(--muted)]">
              consensus {latest.consensus_score.toFixed(2)}
            </span>
          ) : null}
        </div>
        <p className="text-xs text-[var(--text)]">{verdict.rationale}</p>
      </div>

      <div className="space-y-2">
        {entries.length ? entries.map(entry => (
          <DebateArg key={`${entry.role}-${entry.argument.slice(0, 24)}`} entry={entry} />
        )) : (
          <div className="ui-empty p-3 text-xs">
            Debate payload did not include specialist arguments.
          </div>
        )}
      </div>
    </div>
  )
}
