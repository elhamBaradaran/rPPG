import Link from "next/link";
import { getDashboard, fmt, methodColour } from "@/lib/data";
import {
  BarCell,
  Callout,
  Cell,
  ChartNote,
  Chip,
  DefList,
  Dot,
  Grid,
  PageHead,
  Panel,
  Row,
  Stat,
  Table,
} from "@/components/ui";
import type { Hypothesis, MotionCondition, Num } from "@/lib/types";
import { DoseResponse, DriftBars, type DriftRow, type FloorLine } from "./charts";

/** Short readable names, and what the person was actually asked to do. */
const NAME: Record<string, string> = {
  still: "still",
  talk: "talking",
  full: "mixed",
  slow: "slow turns",
  fast: "fast turns",
  lean: "leaning",
};
const CUE: Record<string, string> = {
  still: "nothing at all - the control",
  talk: "speak, head kept still",
  full: "turn, nod and lean at once",
  slow: "turn left and right, one cycle per 4 s",
  fast: "the same turn, one cycle per 1.5 s",
  lean: "move closer to the camera and back",
};

function faceFraction(c: MotionCondition): number | null {
  const d = c.displacement_max_px;
  const w = c.face_width_px;
  if (d === null || w === null || !w) return null;
  return (d / w) * 100;
}

/** Evidence keys are machine names; give each a plain label and its own units. */
const EV_LABEL: Record<string, string> = {
  max_displacement_fraction_of_face_width: "Furthest the face ever moved",
  displacement_error_correlation: "Does error track the face moving?",
  talk_displacement_px: "Face moved while talking",
  talk_drift: "Error while talking",
  still_drift: "Error while still",
};

function evValue(key: string, v: Num): string {
  if (v === null || Number.isNaN(v)) return "—";
  if (key.includes("fraction")) return `${(v * 100).toFixed(0)} % of a face width`;
  if (key.includes("correlation")) return `${v.toFixed(2)} out of 1.00 - no link`;
  if (key.endsWith("_px")) return `${v.toFixed(1)} px`;
  return `${v.toFixed(1)} BPM`;
}

/** A small figure inside a hypothesis panel - the shape of Stat, at panel scale. */
function Fig({
  label,
  value,
  unit,
  tone = "neutral",
}: {
  label: string;
  value: string;
  unit: string;
  tone?: "neutral" | "bad" | "phase" | "pos";
}) {
  const box = {
    neutral: "border-line bg-bg-raised",
    bad: "border-bad/40 bg-bad/[0.08]",
    phase: "border-phase/30 bg-phase/[0.06]",
    pos: "border-pos/30 bg-pos/[0.06]",
  }[tone];
  const ink = {
    neutral: "text-ink",
    bad: "text-bad",
    phase: "text-phase",
    pos: "text-pos",
  }[tone];
  return (
    <div className={`rounded-md border px-3 py-2 ${box}`}>
      <div className="text-2xs leading-snug text-ink-faint">{label}</div>
      <div className="mt-0.5 flex items-baseline gap-1">
        <span className={`num text-xl font-semibold ${ink}`}>{value}</span>
        <span className="text-2xs text-ink-faint">{unit}</span>
      </div>
    </div>
  );
}

export default async function MotionProtocol() {
  const d = await getDashboard();
  const m = d.motion;

  if (!m) {
    return (
      <>
        <PageHead eyebrow="Motion protocol" title="What actually breaks it?" />
        <Panel>
          <p className="text-sm text-ink-soft">
            The motion protocol is not in this build of the data. Run{" "}
            <code className="font-mono text-xs">15_motion_protocol.py</code> and re-export{" "}
            <code className="font-mono text-xs">results/dashboard.json</code>.
          </p>
        </Panel>
      </>
    );
  }

  const conditions = [...m.conditions].sort(
    (a, b) => (a.motion_dose ?? 0) - (b.motion_dose ?? 0),
  );
  const methods = Array.from(new Set(conditions.flatMap((c) => Object.keys(c.methods))));

  const rows: DriftRow[] = conditions.map((c) => {
    const drifts = methods.map((n) => c.methods[n]?.drift ?? null);
    const r: DriftRow = {
      label: NAME[c.label] ?? c.label,
      // The control's measured movement is a hair below its own baseline; it is zero by
      // construction, so it is pinned to zero rather than drawn as negative movement.
      dose: Math.max(0, c.motion_dose ?? 0),
      top: Math.max(...drifts.map((v) => v ?? 0)),
    };
    for (const n of methods) r[n] = c.methods[n]?.drift ?? null;
    return r;
  });

  const floors: FloorLine[] = Object.entries(m.noise_floor)
    .filter(([, v]) => typeof v === "number")
    .map(([name, v]) => ({ name, value: v as number }));

  const byLabel = (l: string) => conditions.find((c) => c.label === l);
  const still = byLabel("still");
  const slow = byLabel("slow");
  const fast = byLabel("fast");
  const primary = methods[0] ?? "";
  const stillRange = still && primary ? still.methods[primary]?.range : undefined;

  // The single worst condition/method pair, for the headline stat and every bar scale.
  let worst = { drift: 0, label: "", method: "" };
  for (const c of conditions) {
    for (const n of methods) {
      const v = c.methods[n]?.drift;
      if (typeof v === "number" && v > worst.drift)
        worst = { drift: v, label: NAME[c.label] ?? c.label, method: n };
    }
  }
  const maxDrift = Math.max(worst.drift, 1);

  // Tripling the movement between the two turning conditions changes nothing.
  const doseRatio =
    slow?.motion_dose && fast?.motion_dose ? fast.motion_dose / slow.motion_dose : null;
  const driftShift =
    slow && fast && primary
      ? (fast.methods[primary]?.drift ?? 0) - (slow.methods[primary]?.drift ?? 0)
      : null;

  const clean = d.headline.held_out_mae_vs_reference;
  const floorValue = floors[0]?.value ?? null;
  const gap = floorValue && clean ? floorValue / clean : null;
  const widestFraction = conditions.reduce<number | null>((acc, c) => {
    const f = faceFraction(c);
    return f === null ? acc : acc === null ? f : Math.max(acc, f);
  }, null);

  const hyps = m.hypotheses_tested ?? [];

  return (
    <>
      <PageHead eyebrow="Motion protocol" title="What actually breaks it?">
        Two earlier recordings disagreed about whether movement breaks the measurement,
        because nobody had measured how much movement there was. So the test was rebuilt:
        five conditions, each recorded as still, then the condition, then still again, so
        every one carries its own baseline — and the movement was measured, not described.
      </PageHead>

      <Grid cols={4}>
        {floors.map((f) => (
          <Stat
            key={f.name}
            label={`${f.name} · at rest`}
            value={fmt(f.value, 1)}
            unit="BPM off"
            tone="warn"
            meter={f.value}
            meterMax={maxDrift}
            hint="Error while sitting perfectly still - nothing is moving."
          />
        ))}
        <Stat
          label="Worst condition"
          value={fmt(worst.drift, 1)}
          unit="BPM off"
          tone="bad"
          meter={worst.drift}
          meterMax={maxDrift}
          hint={`${worst.method} during ${worst.label}. About twice the resting error, and nothing gets worse than this.`}
        />
        <Stat
          label="Slow turns → fast turns"
          value={doseRatio ? `${doseRatio.toFixed(1)}×` : "—"}
          unit="more movement"
          hint={`Triple the movement, and the error changes by ${fmt(
            driftShift === null ? null : Math.abs(driftShift),
            1,
          )} BPM.`}
        />
      </Grid>

      <Panel
        title="Movement doubles the error, then stops mattering"
        hint="Each dot is one condition. Movement is how much the picture inside the face crop changes between frames, above what it changes while still."
      >
        <DoseResponse rows={rows} methods={methods} plateauFrom={slow?.motion_dose ?? undefined} />
        <ChartNote>
          The curve lifts off the resting floor to roughly double it, then flattens. Going
          from slow to fast head turns triples the measured movement and changes nothing,
          so how much someone moves is not what sets the ceiling.
        </ChartNote>
      </Panel>

      <Panel
        title="Which conditions differ from doing nothing?"
        hint="The same numbers as bars. Each dashed line is that method's error while still, so a bar level with its own line added nothing measurable."
      >
        <DriftBars rows={rows} methods={methods} floors={floors} />
        <ChartNote>
          Talking sits on the line. Only turning the head lifts either method clearly above
          its own floor — and both rise together, so neither is the robust one.
        </ChartNote>
      </Panel>

      <Panel
        title="Every condition, side by side"
        hint="Resting rate comes from the still half of the same recording. Drift is how far the reading strayed from that baseline."
        right={<span className="font-mono">still → condition → still</span>}
      >
        <Table
          head={[
            "Condition",
            "Movement",
            "Face moved",
            "of face",
            ...methods.flatMap((n) => [
              <span key={n} className="inline-flex items-center gap-1.5">
                <Dot colour={methodColour(n)} />
                {n} rest
              </span>,
              "during",
              "drift",
            ]),
          ]}
          align={["l", "r", "r", "r", ...methods.flatMap(() => ["r", "r", "r"] as const)]}
        >
          {conditions.map((c) => (
            <Row key={c.label} highlight={c.label === "still"}>
              <Cell>
                <div className="font-medium text-ink">{NAME[c.label] ?? c.label}</div>
                <div className="text-2xs text-ink-faint">{CUE[c.label] ?? "—"}</div>
              </Cell>
              <Cell num>{fmt(Math.max(0, c.motion_dose ?? 0), 2)}</Cell>
              <Cell num>{fmt(c.displacement_max_px, 0, " px")}</Cell>
              <Cell num muted>{fmt(faceFraction(c), 0, " %")}</Cell>
              {methods.map((n) => {
                const s = c.methods[n];
                return [
                  <Cell key={`${n}-b`} num muted>
                    {fmt(s?.baseline_hr ?? null, 0)}
                  </Cell>,
                  <Cell key={`${n}-c`} num>
                    {fmt(s?.condition_hr ?? null, 0)}
                  </Cell>,
                  <BarCell
                    key={`${n}-d`}
                    value={s?.drift ?? null}
                    max={maxDrift}
                    colour={methodColour(n)}
                    digits={1}
                  />,
                ];
              })}
            </Row>
          ))}
        </Table>
        <ChartNote>
          The face never travels far: the largest excursion anywhere is{" "}
          <span className="num">{fmt(widestFraction, 0, " %")}</span> of a face width, so it
          stays inside the crop throughout. And even in the control, single readings from{" "}
          {primary} span{" "}
          <span className="num">
            {fmt(stillRange?.[0] ?? null, 0)}–{fmt(stillRange?.[1] ?? null, 0)}
          </span>{" "}
          BPM while nothing at all is happening.
        </ChartNote>
      </Panel>

      <div>
        <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h2>Three explanations, all rejected</h2>
          <span className="text-xs text-ink-faint">
            each was turned into a test that could kill it
          </span>
        </div>

        <Grid cols={3}>
          {hyps.map((h: Hypothesis, i: number) => {
            const ev = h.evidence ?? {};
            const fixed = ev.drift_static ?? null;
            const tracked = ev.drift_dynamic ?? null;
            const isTracker = fixed !== null && tracked !== null;
            const isTalk = ev.talk_drift !== undefined && ev.still_drift !== undefined;
            const shown = new Set(
              isTracker
                ? ["drift_static", "drift_dynamic"]
                : isTalk
                  ? ["talk_drift", "still_drift"]
                  : [],
            );
            const rest = Object.entries(ev).filter(([k]) => !shown.has(k));
            const ratio = isTracker && fixed ? tracked! / fixed : null;

            return (
              <Panel key={i} title={h.hypothesis} hint={`${h.test}.`}>
                <div className="-mt-1 mb-3">
                  <Chip tone="bad">{h.result}</Chip>
                </div>

                {isTracker && (
                  <>
                    <div className="grid grid-cols-2 gap-2">
                      <Fig
                        label="Crop fixed on the first frame"
                        value={fmt(fixed, 1)}
                        unit="BPM off"
                      />
                      <Fig
                        label="Face re-detected every second"
                        value={fmt(tracked, 1)}
                        unit="BPM off"
                        tone="bad"
                      />
                    </div>
                    <div className="mt-2 rounded-md border border-bad/40 bg-bad/[0.08] px-3 py-2.5 text-xs leading-relaxed text-ink-soft">
                      <span className="num text-sm font-semibold text-bad">
                        {fmt(ratio, 1)}× worse
                      </span>{" "}
                      — following the face made it worse, not better. A box that is
                      re-detected every second jitters, and that jitter is movement of its
                      own.
                    </div>
                  </>
                )}

                {isTalk && (
                  <div className="grid grid-cols-2 gap-2">
                    <Fig
                      label="Talking, head still"
                      value={fmt(ev.talk_drift ?? null, 1)}
                      unit="BPM off"
                    />
                    <Fig
                      label="Doing nothing"
                      value={fmt(ev.still_drift ?? null, 1)}
                      unit="BPM off"
                    />
                  </div>
                )}

                {!isTracker && !isTalk && slow && fast && (
                  <div className="grid grid-cols-2 gap-2">
                    {methods.map((n) => (
                      <Fig
                        key={n}
                        label={`${n}, slow then fast turns`}
                        value={`${fmt(slow.methods[n]?.drift ?? null, 1)} → ${fmt(
                          fast.methods[n]?.drift ?? null,
                          1,
                        )}`}
                        unit="BPM off"
                        tone={n.toLowerCase().includes("phase") ? "phase" : "pos"}
                      />
                    ))}
                  </div>
                )}

                {rest.length > 0 && (
                  <div className="mt-4 border-t border-line-soft pt-3">
                    <DefList
                      items={rest.map(([k, v]): [string, string] => [
                        EV_LABEL[k] ?? k.replace(/_/g, " "),
                        evValue(k, v),
                      ])}
                    />
                  </div>
                )}

                {h.note && (
                  <p className="mt-4 border-l-2 border-bad/40 pl-3 text-sm leading-relaxed text-ink-soft">
                    {h.note}
                  </p>
                )}
              </Panel>
            );
          })}
        </Grid>
      </div>

      <Callout tone="warn" title="What survived: the twenty-fold gap">
        <p>
          The same model is about <span className="num text-ink">{fmt(clean)} BPM</span> off
          on controlled recordings and about{" "}
          <span className="num text-warn">{fmt(floorValue, 0)} BPM</span> off on this webcam
          before anyone moves
          {gap !== null && (
            <>
              {" "}
              — roughly <span className="num text-ink">{fmt(gap, 0)}×</span> worse
            </>
          )}
          . Movement adds a further doubling on top of that, and then stops.
        </p>
        <p>
          So the bottleneck is the camera, the lighting, the distance and how much of the
          frame the face fills — not movement, and not the model. A better capture setup is
          worth more than a more motion-tolerant architecture.{" "}
          <Link href="/validation" className="text-warn underline underline-offset-4">
            The controlled-data figure →
          </Link>
        </p>
      </Callout>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line pt-4 text-2xs text-ink-dim">
        <span>
          Five recordings, one person, one webcam. {fmt(m.window_s, 0)}-second windows;
          smartwatch cross-check {fmt(m.watch_bpm, 0)} BPM.
        </span>
        <span>·</span>
        <span className="font-mono">
          13_record_full.py · 14_static_vs_dynamic.py · 15_motion_protocol.py
        </span>
      </div>
    </>
  );
}
