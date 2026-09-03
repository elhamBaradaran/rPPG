// Copy the Python-generated results into public/ so the static site can fetch them.
//
// The dashboard never computes anything: every number it shows was produced by a script
// in Models/PHASE-Net and written to results/. This step just makes those files reachable
// from the browser. It runs automatically before `npm run dev` and `npm run build`.

import { cp, mkdir, readdir, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const results = path.resolve(here, "..", "..", "results");
const target = path.resolve(here, "..", "public", "data");

if (!existsSync(results)) {
  console.error(`No results folder at ${results}`);
  console.error("Run:  python Models/PHASE-Net/export_dashboard.py");
  process.exit(1);
}

await mkdir(target, { recursive: true });
await cp(results, target, { recursive: true });

const files = await readdir(target);
const waveforms = existsSync(path.join(target, "waveforms"))
  ? (await readdir(path.join(target, "waveforms"))).length
  : 0;
const size = (await stat(path.join(target, "dashboard.json"))).size;

console.log(
  `synced ${files.length} entries to public/data ` +
    `(dashboard.json ${(size / 1024).toFixed(0)} KB, ${waveforms} waveforms)`
);
