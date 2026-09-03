import type { Dashboard, Waveform } from "./types";

// Re-exported so server components can keep importing everything from "@/lib/data".
// Client components must import them from "@/lib/format" instead - see that file.
export { fmt, PALETTE, methodColour } from "./format";

// With `output: export` every page is rendered at build time, so the JSON is read from
// disk during the build rather than fetched by the browser. Waveforms stay separate and
// are fetched on demand - all fifteen together are larger than the rest of the payload.

// Must carry the NEXT_PUBLIC_ prefix: only those variables are inlined into the browser
// bundle at build time. A plain process.env.BASE_PATH reads as undefined in client code,
// which would silently break waveform fetches when the site is served from a sub-path.
const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";

export async function getDashboard(): Promise<Dashboard> {
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  const file = path.join(process.cwd(), "public", "data", "dashboard.json");
  return JSON.parse(await fs.readFile(file, "utf-8")) as Dashboard;
}

/** Client-side: fetch one subject's waveform and spectrum. */
export async function fetchWaveform(relPath: string): Promise<Waveform> {
  const res = await fetch(`${BASE}/data/${relPath}`);
  if (!res.ok) throw new Error(`waveform not found: ${relPath}`);
  return (await res.json()) as Waveform;
}

