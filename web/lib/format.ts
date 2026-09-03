// Display helpers that are safe in the browser.
//
// These live apart from lib/data.ts on purpose: that module reads dashboard.json off the
// filesystem, so it pulls in node:fs and node:path, and webpack cannot bundle those for a
// client component. Charts are client components and need the palette and the formatter,
// so anything they touch belongs here. lib/data.ts re-exports all three, which keeps the
// server pages importing from a single place.

/** Format a number for display, tolerating the nulls the exporter emits for NaN. */
export function fmt(v: number | null | undefined, digits = 2, suffix = ""): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v.toFixed(digits)}${suffix}`;
}

/**
 * Chart colours for the dark panel theme, matching tailwind.config.ts.
 *
 * Teal is always PHASE-Net and amber is always POS, on every chart on every page, so the
 * mapping only has to be learnt once.
 */
export const PALETTE = {
  phasenet: "#2dd4bf",
  posRef: "#fbbf24",
  posOurs: "#b45309",
  reference: "#818cf8",
  device: "#c084fc",
  grid: "#1e2b45",
  axis: "#6b7f9e",
  good: "#4ade80",
  warn: "#fb923c",
  bad: "#f87171",
  tooltipBg: "#111a2e",
} as const;

/** Stable colour per method name, so every chart agrees. */
export function methodColour(name: string): string {
  const n = name.toLowerCase();
  if (n.includes("phase")) return PALETTE.phasenet;
  if (n.includes("ours")) return PALETTE.posOurs;
  if (n.includes("pos")) return PALETTE.posRef;
  return PALETTE.reference;
}

/** Shared Recharts tooltip styling, so every chart's hover card looks the same. */
export const TOOLTIP = {
  contentStyle: {
    background: PALETTE.tooltipBg,
    border: "1px solid #2c3e60",
    borderRadius: 8,
    fontSize: 12,
    color: "#e6edf7",
    boxShadow: "0 8px 24px -12px rgba(0,0,0,0.8)",
  },
  labelStyle: { color: "#9db0cc", fontSize: 11, marginBottom: 2 },
  itemStyle: { color: "#e6edf7", fontSize: 12 },
} as const;

export const AXIS = {
  tick: { fill: PALETTE.axis, fontSize: 11 },
  stroke: PALETTE.grid,
} as const;
