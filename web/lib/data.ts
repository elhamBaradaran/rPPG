import type { Dashboard, Waveform } from "./types";

// Re-exported so server components can keep importing everything from "@/lib/data".
// Client components must import them from "@/lib/format" instead - see that file.
export { fmt, PALETTE, methodColour } from "./format";

// With `output: export` every page is rendered at build time, so the JSON is read from
// disk during the build rather than fetched by the browser. Waveforms stay separate and
// are fetched on demand - all fifteen together are larger than the rest of the payload.

const BASE = process.env.BASE_PATH || "";

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

