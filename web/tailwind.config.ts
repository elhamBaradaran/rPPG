import type { Config } from "tailwindcss";

/**
 * Dark instrument-panel palette. One place to restyle the whole dashboard.
 *
 * The two method colours are fixed and used everywhere - teal is always PHASE-Net,
 * amber is always POS - so a reader learns the mapping once and it holds on every chart.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: { DEFAULT: "#070b17", raised: "#0d1425", panel: "#111a2e" },
        line: { DEFAULT: "#1e2b45", soft: "#16223a", bright: "#2c3e60" },
        ink: { DEFAULT: "#e6edf7", soft: "#9db0cc", faint: "#6b7f9e", dim: "#4a5b76" },
        // method identities - never reuse these for anything else
        phase: { DEFAULT: "#2dd4bf", dim: "#134e4a", glow: "#5eead4" },
        pos: { DEFAULT: "#fbbf24", dim: "#4a3410", glow: "#fcd34d" },
        // semantic
        good: { DEFAULT: "#4ade80", dim: "#14361f" },
        warn: { DEFAULT: "#fb923c", dim: "#43220c" },
        bad: { DEFAULT: "#f87171", dim: "#3f1618" },
        info: { DEFAULT: "#818cf8", dim: "#1e1b4b" },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.6)",
      },
      backgroundImage: {
        "grid-fade":
          "radial-gradient(ellipse 80% 50% at 50% -20%, rgba(45,212,191,0.10), transparent)",
      },
    },
  },
  plugins: [],
};

export default config;
