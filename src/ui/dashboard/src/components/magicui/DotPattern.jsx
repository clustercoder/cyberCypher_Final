export default function DotPattern({ className = '' }) {
  return (
    <div
      aria-hidden="true"
      className={`dot-pattern pointer-events-none absolute inset-0 ${className}`}
    />
  )
}
