import { useEffect, useRef } from 'react'

const PHASE_COLOR = {
  observe: 'text-[var(--accent)]',
  reason: 'text-[var(--accent-2)]',
  decide: 'text-[var(--high)]',
  debate: 'text-[#c68af7]',
  verify: 'text-[#84caff]',
  act: 'text-[var(--ok)]',
  learn: 'text-[#47B5B8]',
  start: 'text-[var(--ok)]',
  stop: 'text-[var(--muted)]',
  inject: 'text-[var(--high)]',
  kill_switch: 'text-[var(--critical)]',
  error: 'text-[var(--critical)]',
}

const PHASE_DOT = {
  observe: 'bg-[var(--accent)]',
  reason: 'bg-[var(--accent-2)]',
  decide: 'bg-[var(--high)]',
  debate: 'bg-[#c68af7]',
  verify: 'bg-[#84caff]',
  act: 'bg-[var(--ok)]',
  learn: 'bg-[#47B5B8]',
  start: 'bg-[var(--ok)]',
  stop: 'bg-[var(--muted)]',
  inject: 'bg-[var(--high)]',
  kill_switch: 'bg-[var(--critical)]',
  error: 'bg-[var(--critical)]',
}

const SEVERITY_BADGE = {
  critical: 'badge-critical',
  high: 'badge-high',
  medium: 'badge-medium',
  low: 'badge-low',
}

function EventRow({ event, isNew }) {
  const phase = event.phase || 'observe'
  const color = PHASE_COLOR[phase] || 'text-[var(--text)]'
  const dot = PHASE_DOT[phase] || 'bg-[var(--muted)]'
  const time = (() => {
    if (!event.ts) return '--:--:--'
    const date = new Date(event.ts)
    return Number.isNaN(date.getTime()) ? '--:--:--' : date.toLocaleTimeString()
  })()
  const isKill = phase === 'kill_switch'

  return (
    <div
      className={`ui-row grid grid-cols-[auto_auto_auto_1fr_auto] items-start gap-2 py-1.5 text-xs ${
        isNew ? 'event-row-new' : ''
      } ${isKill ? 'rounded bg-[rgba(255,93,108,0.11)] px-1' : ''}`}
    >
      <span
        className={`mt-1.5 h-1.5 w-1.5 rounded-full ${dot}`}
      />
      <span className="mono w-[72px] text-[10px] text-[var(--muted)]">{time}</span>
      <span className={`mono w-[62px] text-[10px] uppercase ${color}`}>{phase}</span>
      <span className={`min-w-0 break-words ${isKill ? 'font-semibold text-[var(--critical)]' : 'text-[var(--text)]'}`}>
        {event.message}
      </span>
      {event.anomalies?.length > 0 ? (
        <span className={SEVERITY_BADGE[event.anomalies[0]?.severity] || 'badge-low'}>
          {event.anomalies.length}
        </span>
      ) : null}
    </div>
  )
}

export default function AgentFeed({ events = [] }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'auto' })
  }, [events.length])

  const lastIndex = events.length - 1

  return (
    <div className="flex h-[280px] flex-col panel-live">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="ui-title flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full bg-[var(--accent)]" />
          Agent Activity
        </h2>
        <span className="ui-pill">
          {events.length}
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        {events.length === 0 ? (
          <div className="py-6 text-center text-xs text-[var(--muted)]">
            Waiting for agent events…
          </div>
        ) : (
          events.map((event, index) => (
            <EventRow
              key={`${event.ts || index}-${event.phase || 'event'}`}
              event={event}
              isNew={index === lastIndex}
            />
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
