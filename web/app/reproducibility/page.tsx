import { getDashboard, fmt, PALETTE } from "@/lib/data";
import {
  BarCell,
  Callout,
  ChartNote,
  Chip,
  Cell,
  DefList,
  Dot,
  Grid,
  PageHead,
  Panel,
  Row,
  Stat,
  Table,
} from "@/components/ui";

// The exporter writes these blocks as loose dictionaries, so the shapes are narrowed here
// rather than in types.ts - only this page needs them.
type ModelTrace = {
  name?: string;
  variant?: string;
  weights_file?: string;
  weights_sha256?: string;
  params_total?: number;
  params_inference_path?: number;
  params_claimed_in_paper?: number;
  paper?: { venue?: string };
};

/** Any value out of the export, rendered as text, with the exporter's nulls handled. */
function s(v: unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  return String(v);
}

/** Millions, to two places - the unit every parameter count on this page is quoted in. */
function m(n: number | null | undefined): string {
  return n === null || n === undefined ? "—" : (n / 1e6).toFixed(2);
}

const PIPELINE: [string, string][] = [
  ["01_load_phasenet.py", "Loads the released weights; every tensor matches the model exactly."],
  ["04_ubfc_eval.py", "Scores the model against the dataset and writes results/*.json."],
  ["05_cache_signals.py", "Caches the model output, so extraction tests re-run in seconds instead of reprocessing 24 GB of video."],
  ["08_windowed_eval.py", "The evaluation protocol: ten-second windows, one-second step."],
  ["09_pos_baseline.py", "Runs the POS baseline on identical face crops."],
  ["10_compare_models.py", "Scores both methods under that single protocol."],
  ["13_record_full.py", "Records the webcam sessions used for the motion work."],
  ["14_static_vs_dynamic.py", "Fixed crop against per-frame face tracking, same recording."],
  ["15_motion_protocol.py", "The controlled protocol: still, slow, fast, talking."],
  ["export_dashboard.py", "Assembles the one JSON file this site reads."],
];

export default async function Provenance() {
  const d = await getDashboard();
  const run = d.traceability.run;
  const env = (run?.environment ?? {}) as Record<string, unknown>;
  const model = (d.traceability.model ?? {}) as ModelTrace;
  const bench = d.extraction_benchmark;
  const val = d.validation;

  const pTotal = model.params_total ?? null;
  const pInf = model.params_inference_path ?? null;
  const pPaper = model.params_claimed_in_paper ?? null;
  const pTrain = pTotal !== null && pInf !== null ? pTotal - pInf : null;
  const pRatio = pInf !== null && pPaper ? pInf / pPaper : null;

  const pre = (val?.preprocessing ?? {}) as Record<string, unknown>;
  const proto = (val?.metrics_windowed?.protocol ?? {}) as Record<string, unknown>;
  const nHeld = val?.metrics_windowed?.held_out?.n_subjects ?? null;

  // Rows stay in the exporter's order. One shared bar scale across all three numeric
  // columns, so the gap between the loose and the strict reference is visible as length.
  const rows = (bench?.methods ?? []).map((x) => {
    const ref = x.vs_reference.mae_held_out;
    const dev = x.vs_device.mae_held_out;
    return {
      method: x.method,
      ref,
      dev,
      refErr: x.vs_device.reference_error,
      factor: ref !== null && dev !== null && ref > 0 ? dev / ref : null,
    };
  });
  const barMax = Math.max(
    1,
    ...rows.flatMap((r) => [r.ref, r.dev, r.refErr].filter((v): v is number => v !== null)),
  );
  const worst = rows.reduce<(typeof rows)[number] | null>(
    (a, b) => (b.factor !== null && (a === null || b.factor > (a.factor ?? 0)) ? b : a),
    null,
  );

  return (
    <>
      <PageHead eyebrow="Provenance" title="How were these numbers made?">
        A result nobody can trace is not a result. Every figure on this site is produced by
        a numbered Python script and written to a file; this site only draws it.
      </PageHead>

      <Grid cols={2}>
        <Panel
          title="Provenance"
          hint="Which weights file, which commit, which moment."
          right={<Chip tone="good">inference only</Chip>}
        >
          <DefList
            items={[
              ["Run id", s(run?.id)],
              ["Created (UTC)", run?.created_utc ? new Date(run.created_utc).toUTCString() : "—"],
              [
                "Git commit",
                run?.git_commit ? (
                  <a
                    className="text-phase underline underline-offset-2"
                    href={`${d.project.repository}/commit/${run.git_commit}`}
                  >
                    {run.git_commit}
                  </a>
                ) : (
                  "—"
                ),
              ],
              ["Weights file", s(model.weights_file)],
              [
                "Weights SHA-256",
                <span key="sha" className="block font-mono text-[11px] leading-relaxed break-all">
                  {s(model.weights_sha256)}
                </span>,
              ],
              ["Model", `${s(model.name)} · ${s(model.variant)}`],
            ]}
          />
        </Panel>

        <Panel
          title="Environment"
          hint="Library versions drift and change results quietly, so they are recorded, not assumed."
        >
          <DefList
            items={[
              ["Python", s(env.python)],
              ["PyTorch", s(env.torch)],
              ["NumPy", s(env.numpy)],
              ["SciPy", s(env.scipy)],
              ["OpenCV", s(env.cv2)],
              ["Platform", s(env.platform)],
              ["Device", s(env.device)],
              ["GPU", s(env.gpu)],
              ["GPU memory", env.gpu_memory_gb !== undefined ? `${s(env.gpu_memory_gb)} GB` : "—"],
            ]}
          />
        </Panel>
      </Grid>

      <Panel
        title="Parameter count"
        hint="Counted from the released weights file. The paper's efficiency claim rests on this number."
      >
        <Grid cols={3}>
          <Stat
            label="Checkpoint total"
            value={m(pTotal)}
            unit="M"
            meter={pTotal}
            meterMax={pTotal ?? 1}
            hint="Everything stored in the released file."
          />
          <Stat
            label="Training only"
            value={m(pTrain)}
            unit="M"
            meter={pTrain}
            meterMax={pTotal ?? 1}
            hint="Helps the model learn; never runs when measuring a heart rate."
          />
          <Stat
            label="Inference path"
            value={m(pInf)}
            unit="M"
            tone="warn"
            meter={pInf}
            meterMax={pTotal ?? 1}
            compare={{ value: `${m(pPaper)} M`, label: "stated in the paper" }}
            hint="What actually runs. Bars share one scale."
          />
        </Grid>
        <p className="mt-4 max-w-3xl text-sm leading-relaxed text-ink-soft">
          The training-only part never runs at inference, so excluding it is fair. What
          remains is still about{" "}
          <span className="num text-warn">{pRatio !== null ? pRatio.toFixed(1) : "—"}×</span>{" "}
          the published figure. An observation, not an accusation: the counts may simply
          differ in what each side counted.
        </p>
        <ChartNote>
          No accuracy figure on this site depends on it — inference uses the released
          weights unchanged. The same file also carries four temporal layers where the
          paper&apos;s own ablation settles on three.
        </ChartNote>
      </Panel>

      <Panel
        title="Pipeline"
        hint="Each step writes its own file, so any figure traces back to the script that made it without re-running the ones before."
      >
        <ol className="grid gap-x-8 gap-y-2.5 sm:grid-cols-2">
          {PIPELINE.map(([script, what], i) => (
            <li key={script} className="flex gap-3">
              <span className="num mt-0.5 shrink-0 text-2xs text-ink-dim">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="min-w-0 text-sm">
                <span className="font-mono text-[13px] text-phase">{script}</span>
                <span className="block text-2xs leading-snug text-ink-faint">{what}</span>
              </span>
            </li>
          ))}
        </ol>

        <div className="mt-5 border-t border-line-soft pt-4">
          <div className="mb-3 text-2xs uppercase tracking-wider text-ink-faint">
            Configuration as it ran
          </div>
          <DefList
            items={[
              ["Face detection", s(pre.face_detection)],
              [
                "Input size",
                Array.isArray(pre.resize) ? (pre.resize as number[]).join(" × ") : "—",
              ],
              ["Pixel range", s(pre.pixel_range)],
              ["Clip length", s(pre.clip_length)],
              ["Window / step", `${s(proto.win)} s / ${s(proto.step)} s`],
              ["Spectral estimator", s(proto.estimator)],
            ]}
          />
        </div>
      </Panel>

      {bench && (
        <Panel
          title="Reading a heart rate off the signal"
          hint={`${rows.length} ways of turning a pulse waveform into one number, average error in BPM. Held-out means the ${
            nHeld ?? "few"
          } subjects the released weights were never trained on.`}
        >
          <Callout tone="warn" title="Scoring a method against itself proves nothing">
            <p className="text-sm">{bench.note}</p>
            {worst && (
              <p className="text-sm">
                <span className="text-ink">{worst.method}</span> scores{" "}
                <span className="num">{fmt(worst.ref)}</span> the loose way and{" "}
                <span className="num">{fmt(worst.dev)}</span> against the device — the same
                method, looking{" "}
                <span className="num text-warn">{worst.factor?.toFixed(0)}×</span> better.
              </p>
            )}
          </Callout>

          <div className="mt-4">
            <Table
              head={[
                "Method",
                <span key="a" className="inline-flex items-center gap-1.5">
                  <Dot colour={PALETTE.reference} />
                  vs reference
                </span>,
                <span key="b" className="inline-flex items-center gap-1.5">
                  <Dot colour={PALETTE.good} />
                  vs the oximeter
                </span>,
                "Inflation",
                <span key="c" className="inline-flex items-center gap-1.5">
                  <Dot colour={PALETTE.warn} />
                  Reference error
                </span>,
              ]}
              align={["l", "r", "r", "r", "r"]}
            >
              {rows.map((r) => (
                <Row key={r.method} highlight={r.method === worst?.method}>
                  <Cell>{r.method}</Cell>
                  <BarCell value={r.ref} max={barMax} colour={PALETTE.reference} />
                  <BarCell value={r.dev} max={barMax} colour={PALETTE.good} />
                  <Cell num muted={(r.factor ?? 0) < 3}>
                    {r.factor !== null ? `${r.factor.toFixed(1)}×` : "—"}
                  </Cell>
                  <BarCell value={r.refErr} max={barMax} colour={PALETTE.warn} />
                </Row>
              ))}
            </Table>
          </div>

          <ChartNote>
            All bars share one scale, so the columns compare directly.{" "}
            <span className="text-ink-soft">vs reference</span> runs the same processing on
            the prediction and on the true pulse — it measures whether a method agrees with
            itself, which is why it looks so good.{" "}
            <span className="text-ink-soft">vs the oximeter</span> compares against the
            device&apos;s own displayed heart rate, which none of this code touches.{" "}
            <span className="text-ink-soft">Reference error</span> is how wrong each method
            is at reading the true pulse&apos;s own heart rate, with no model involved — a
            floor under the middle column.
          </ChartNote>
        </Panel>
      )}

      <Callout tone="warn" title="What traceability does not establish">
        <p className="text-sm">
          These numbers are reproducible. That does not make them general.
        </p>
        <ul className="list-disc space-y-1.5 pl-4 text-xs leading-relaxed">
          {d.limitations.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      </Callout>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line pt-4 text-2xs text-ink-dim">
        <span>Generated {new Date(d.generated_utc).toUTCString()}</span>
        <span>·</span>
        <span>schema {d.schema_version}</span>
        {model.paper?.venue && (
          <>
            <span>·</span>
            <span>{model.paper.venue}</span>
          </>
        )}
      </div>
    </>
  );
}
