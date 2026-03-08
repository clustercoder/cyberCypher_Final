export default function BorderBeam({ className = '' }) {
  return (
    <div className={`pointer-events-none absolute inset-0 overflow-hidden rounded-[inherit] ${className}`}>
      <div className="magic-border-beam" />
    </div>
  )
}
