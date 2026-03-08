export default function MagicCard({ children, className = '' }) {
  return (
    <div className={`relative panel panel-magic ${className}`}>
      {children}
    </div>
  )
}
