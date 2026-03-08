export default function ShimmerButton({
  children,
  className = '',
  variant = 'primary',
  type = 'button',
  ...props
}) {
  return (
    <button
      type={type}
      className={`shimmer-button shimmer-button-${variant} ${className}`}
      {...props}
    >
      <span>{children}</span>
    </button>
  )
}
