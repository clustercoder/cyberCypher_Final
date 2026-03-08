export default function AnimatedGradientText({ children, className = '' }) {
  return (
    <span className={`magic-gradient-text ${className}`}>
      {children}
    </span>
  )
}
