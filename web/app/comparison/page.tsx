import type { ReactNode } from "react";
import { getDashboard, fmt, methodColour } from "@/lib/data";
import {
  BarCell,
  Callout,
  ChartNote,
  Cell,
  Chip,
  Dot,
  Grid,
  PageHead,
  Panel,
  Row,
  Stat,
  Table,
} from "@/components/ui";
import { GroupedBars, RankBars, type GroupedRow, type RankRow, type Series } from "./charts";

/** Friendlier names than the keys the exporter uses. Colours still key off the raw name. */
const LABEL: Record<string, string> = {
  "PHASE-Net": "PHASE-Net",
  "POS ref": "POS reference",
  "POS ours": "POS written here",
};
const label = (m: string) => LABEL[m] ?? m;

/** Largest per-subject error on the page, used as one common scale for every table bar. */
const ERROR_SCALE = 28;

export default async function ComparisonPage() {
  const d = await getDashboard();
  const c = d.comparison;

  if (!c) {
    return (
      <>
        <PageHead eyebrow="Model comparison" title="Is the deep model worth it?">
          This export contains no comparison data.
        </PageHead>
        <Panel>
          <p className="text-sm text-ink-soft">
            Run <code className="font-mono text-xs">10_compare_models.py</code>, then{" "}
            <code className="font-mono text-xs">export_dashboard.py</code>.
          </p>
        </Panel>
      </>
    );
  }

  const { methods, subjects } = c;
  const proto = c.protocol as { win?: number; step?: number };

  const stat = (
    m: string,
    scope: "held_out" | "all",
    metric: "mae_vs_reference" | "mae_vs_device",
    agg: "mean" | "worst",
  ): number | null => c.summary?.[m]?.[scope]?.[metric]?.[agg] ?? null;

  const series: Series[] = methods.map((m) => ({
    key: m,
    name: label(m),
    colour: methodColour(m),
  }));

  const row = (text: string, read: (m: string) => number | null): GroupedRow => ({
    label: text,
    values: Object.fromEntries(methods.map((m): [string, number | null] => [m, read(m)])),
  });

  const headToHead: GroupedRow[] = [
    row("vs reference waveform", (m) => stat(m, "held_out", "mae_vs_reference", "mean")),
    row("vs the oximeter", (m) => stat(m, "held_out", "mae_vs_device", "mean")),
  ];

  const worst: RankRow[] = methods.map((m) => ({
    name: label(m),
    value: stat(m, "all", "mae_vs_reference", "worst"),
    colour: methodColour(m),
    note: `largest error on any one of the ${c.n_all} people`,
  }));

  // Held out first, so the six the model has never seen form one readable block and the
  // nine it trained on sit faded behind them.
  const ordered = [...subjects].sort(
    (a, b) => Number(b.held_out) - Number(a.held_out) || a.id.localeCompare(b.id),
  );
  const shortId = (id: string) => id.replace(/^subject/, "");
  const perSubject: GroupedRow[] = ordered.map((s) => ({
    label: shortId(s.id),
    values: Object.fromEntries(
      methods.map((m): [string, number | null] => [
        m,
        s.methods?.[m]?.mae_vs_reference ?? null,
      ]),
    ),
  }));
  const trainedOn = new Set(ordered.filter((s) => !s.held_out).map((s) => shortId(s.id)));

  const phaseDev = stat("PHASE-Net", "held_out", "mae_vs_device", "mean");
  const refDev = stat("POS ref", "held_out", "mae_vs_device", "mean");
  const oursDev = stat("POS ours", "held_out", "mae_vs_device", "mean");
  const phaseRef = stat("PHASE-Net", "held_out", "mae_vs_reference", "mean");
  const refRef = stat("POS ref", "held_out", "mae_vs_reference", "mean");
  const oursRef = stat("POS ours", "held_out", "mae_vs_reference", "mean");
  const phaseWorst = stat("PHASE-Net", "all", "mae_vs_reference", "worst");
  const refWorst = stat("POS ref", "all", "mae_vs_reference", "worst");
  const phaseHeldWorst = stat("PHASE-Net", "held_out", "mae_vs_reference", "worst");
  const refHeldWorst = stat("POS ref", "held_out", "mae_vs_reference", "worst");

  const times = phaseRef && refRef ? (refRef / phaseRef).toFixed(1) : "—";
  const oursPenalty = oursDev && refDev ? Math.round((oursDev / refDev) * 100 - 100) : null;

  const head: ReactNode[] = [
    "Subject",
    "Split",
    "Device HR",
    ...methods.flatMap((m): ReactNode[] => [
      <span key={`${m}-h`} className="inline-flex items-center gap-1.5">
        <Dot colour={methodColour(m)} />
        {label(m)} HR
      </span>,
      "vs ref",
      "vs device",
    ]),
  ];
  const align: ("l" | "r")[] = [
    "l",
    "l",
    "r",
    ...methods.flatMap(() => ["r", "r", "r"] as const),
  ];

  return (
    <>
      <PageHead eyebrow="Model comparison" title="Is the deep model worth it?">
        Same {c.n_all} videos, same cropped face, same scoring — the only difference is the
        algorithm: <span className="text-phase">PHASE-Net</span>, the deep model; the{" "}
        <span className="text-pos">reference POS</span> implementation, the standard
        classical baseline; and a <span className="text-pos">simplified POS</span> written
        from scratch here.
      </PageHead>

      <Grid cols={4}>
        <Stat
          label="PHASE-Net"
          value={fmt(phaseDev)}
          unit="BPM off"
          tone="phase"
          meter={phaseDev}
          meterMax={4}
          compare={{ value: fmt(phaseRef), label: "against the reference waveform" }}
          hint={`Average gap from the oximeter — the fingertip clip's own reading — on the ${c.n_held_out} people held out, meaning never used in training.`}
        />
        <Stat
          label="POS reference"
          value={fmt(refDev)}
          unit="BPM off"
          tone="pos"
          meter={refDev}
          meterMax={4}
          compare={{ value: fmt(refRef), label: "against the reference waveform" }}
          hint="Fixed maths, no training. Level with the deep model on the same people."
        />
        <Stat
          label="POS written here"
          value={fmt(oursDev)}
          unit="BPM off"
          tone="pos"
          meter={oursDev}
          meterMax={4}
          compare={{ value: fmt(oursRef), label: "against the reference waveform" }}
          hint={`Same projection maths as above, one detail missing. About ${
            oursPenalty ?? "—"
          } per cent worse.`}
        />
        <Stat
          label={`Worst person, all ${c.n_all}`}
          value={
            <>
              <span className="text-phase">{fmt(phaseWorst, 1)}</span>
              <span className="px-1 text-lg text-ink-dim">vs</span>
              <span className="text-pos">{fmt(refWorst, 1)}</span>
            </>
          }
          unit="BPM"
          hint="Largest error each made on any single person. This, not the average, is the real difference."
        />
      </Grid>

      <Panel
        title="Head to head"
        hint={`Average error in beats per minute on the ${c.n_held_out} held-out people. Lower is better.`}
        right={`${fmt(proto.win, 0)} s windows · ${fmt(proto.step, 0)} s step`}
      >
        <GroupedBars rows={headToHead} series={series} height={300} labels />
        <ChartNote>
          Left is the score papers publish — each method against the pulse trace the finger
          sensor recorded — and the deep model wins by {times}×. Right is the same
          recordings against the heart rate the device reports itself, untouched by our
          code, and the three tie. PHASE-Net was trained to reproduce that pulse trace, so
          the left group marks it on its own homework.
        </ChartNote>
      </Panel>

      <Panel
        title="Worst case, and why it is the headline"
        hint={`Largest error any single person produced, across all ${c.n_all}.`}
      >
        <RankBars rows={worst} />
        <ChartNote>
          An average hides the failures that matter: a monitor that is usually right and
          occasionally {fmt(refWorst, 0)} BPM wrong is not usable. Caveat — nine of these
          fifteen sit inside PHASE-Net&apos;s training data; on the held-out six alone it is{" "}
          {fmt(phaseHeldWorst)} against {fmt(refHeldWorst)} BPM, smaller, same direction.
        </ChartNote>
      </Panel>

      <Panel
        title="Where the averages come from"
        hint="Error against the reference pulse trace, one group per person, held-out people first."
        right={
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-sm bg-ink-dim opacity-40" />
            faded = trained on
          </span>
        }
      >
        <GroupedBars
          rows={perSubject}
          series={series}
          height={360}
          angledTicks
          dim={trainedOn}
        />
        <ChartNote>
          Two people set the whole scale: on subjects 25 and 27 both classical methods lose
          the pulse entirely. On everyone else all three sit under 4 BPM.
        </ChartNote>
      </Panel>

      <Panel
        title="Every person, every method"
        hint="Median heart rate and both errors, in beats per minute. Bars share one scale, so any two can be compared by eye."
      >
        <Table head={head} align={align}>
          {ordered.map((s) => (
            <Row key={s.id} highlight={s.held_out}>
              <Cell strong>{s.id.replace(/^subject/, "subject ")}</Cell>
              <Cell>
                {s.held_out ? <Chip tone="phase">held out</Chip> : <Chip>trained on</Chip>}
              </Cell>
              <Cell num muted>
                {fmt(s.device_hr, 0)}
              </Cell>
              {methods.flatMap((m) => {
                const v = s.methods?.[m];
                return [
                  <Cell key={`${m}-hr`} num>
                    {fmt(v?.median_hr ?? null, 1)}
                  </Cell>,
                  <BarCell
                    key={`${m}-ref`}
                    value={v?.mae_vs_reference ?? null}
                    max={ERROR_SCALE}
                    colour={methodColour(m)}
                  />,
                  <BarCell
                    key={`${m}-dev`}
                    value={v?.mae_vs_device ?? null}
                    max={ERROR_SCALE}
                    colour={methodColour(m)}
                  />,
                ];
              })}
            </Row>
          ))}
        </Table>
        <ChartNote>
          On subjects 25 and 27 all three methods differ from the device by about 20 BPM in
          the same direction, and two independent readings of the device&apos;s own waveform
          side with the methods. A broken reference, not three simultaneous failures — and
          both are trained-on, so the held-out figures are untouched.
        </ChartNote>
      </Panel>

      <Grid cols={2}>
        <Callout tone="info" title="Why the usual score flatters the deep model">
          <p className="text-xs leading-relaxed">
            PHASE-Net was trained to reproduce the reference pulse trace, so scoring it
            against that same trace asks how well it hit its own training target.
          </p>
          <p className="text-xs leading-relaxed">
            POS was never trained on anything — it is fixed maths from the optics of light
            through skin. Hence a {times}× gap on the left chart and none against the
            device.
          </p>
        </Callout>

        <Callout
          tone="warn"
          title={`Our POS is ${oursPenalty ?? "—"} per cent worse, and we know why`}
        >
          <p className="text-xs leading-relaxed">
            Both implementations compute the identical projection. The reference applies it
            inside a 1.6 second sliding window; ours fits the whole minute at once.
          </p>
          <p className="text-xs leading-relaxed">
            Lighting and skin appearance drift over a minute and one global fit cannot track
            that. The baseline is not weak because the maths is simple.
          </p>
        </Callout>
      </Grid>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line pt-4 text-2xs text-ink-dim">
        <span>
          <code className="font-mono">10_compare_models.py</code>, cached signals for all{" "}
          {c.n_all} subjects
        </span>
        <span>·</span>
        <span>{c.n_held_out} held out</span>
        <span>·</span>
        <span>generated {new Date(d.generated_utc).toUTCString()}</span>
      </div>
    </>
  );
}
