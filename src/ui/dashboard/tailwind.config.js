/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        cyber: {
          bg: "#0a0e1a",
          panel: "#0f1628",
          border: "#1e2d4a",
          accent: "#00d4ff",
          green: "#00ff88",
          red: "#ff4444",
          yellow: "#ffaa00",
          purple: "#7c3aed",
        },
      },
      keyframes: {
        "pulse-glow-red": {
          "0%, 100%": { boxShadow: "0 0 4px 1px rgba(239,68,68,0.3)" },
          "50%": { boxShadow: "0 0 12px 3px rgba(239,68,68,0.7)" },
        },
        "pulse-glow-amber": {
          "0%, 100%": { boxShadow: "0 0 4px 1px rgba(245,158,11,0.3)" },
          "50%": { boxShadow: "0 0 10px 2px rgba(245,158,11,0.6)" },
        },
        "pulse-glow-cyan": {
          "0%, 100%": { boxShadow: "0 0 4px 1px rgba(0,212,255,0.2)" },
          "50%": { boxShadow: "0 0 10px 2px rgba(0,212,255,0.5)" },
        },
        flicker: {
          "0%, 100%": { opacity: "1" },
          "92%": { opacity: "1" },
          "93%": { opacity: "0.85" },
          "94%": { opacity: "1" },
          "96%": { opacity: "0.9" },
          "97%": { opacity: "1" },
        },
        scan: {
          "0%": { backgroundPosition: "0 0" },
          "100%": { backgroundPosition: "0 100vh" },
        },
        "slide-in": {
          from: { opacity: "0", transform: "translateX(8px)" },
          to: { opacity: "1", transform: "translateX(0)" },
        },
        "flash-value": {
          "0%": { color: "#00d4ff", textShadow: "0 0 8px #00d4ff" },
          "100%": { color: "inherit", textShadow: "none" },
        },
        "spin-slow": {
          from: { transform: "rotate(0deg)" },
          to: { transform: "rotate(360deg)" },
        },
        "blink-dot": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.2" },
        },
      },
      animation: {
        "pulse-glow-red": "pulse-glow-red 1.5s ease-in-out infinite",
        "pulse-glow-amber": "pulse-glow-amber 2s ease-in-out infinite",
        "pulse-glow-cyan": "pulse-glow-cyan 3s ease-in-out infinite",
        "flicker": "flicker 8s linear infinite",
        "slide-in": "slide-in 0.18s ease-out",
        "flash-value": "flash-value 0.6s ease-out",
        "spin-slow": "spin-slow 8s linear infinite",
        "blink-dot": "blink-dot 1.2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
}
