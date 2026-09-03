import Link from "next/link";
import { getDashboard, fmt, PALETTE } from "@/lib/data";
import {
  Callout,
  ChartNote,
  Chip,
  Grid,
  PageHead,
  Panel,
  Stat,
} from "@/components/ui";
import { ErrorScale, type ScaleRow } from "./charts";

export default async function Overview() {
  const d = await getDashboard();
  const h = d.headline;
  const cmp = d.comparison;

  const scale: ScaleRow[] = [
    {
      name: "Paper's claim",
      value: h.paper_claim,
      colour: PALETTE.reference,
      note: "reported by the PHASE-Net authors on the full test split",
    },
    {
      name: "Measured, clean data",
      value: h.held_out_mae_vs_reference ?? 0,
      colour: PALETTE.phasenet,
      note: "our held-out subjects, same convention as the paper",
    },
    {
      name: "Against the oximeter",
      value: h.held_out_mae_vs_device ?? 0,
      colour: PALETTE.good,
      note: "compared with the device's own reading, which our code never touches",
    },
    {
      name: "This webcam, sitting still",
      value: h.webcam_noise_floor ?? 0,
      colour: PALETTE.warn,
      note: "error with no movement at all - the real bottleneck",
    },
  ];

  const worstPhase = cmp?.summary?.["PHASE-Net"]?.all?.mae_vs_reference?.worst ?? null;
  const worstPos = cmp?.summary?.["POS ref"]?.all?.mae_vs_reference?.worst ?? null;

  return (
    <>
      <PageHead eyebrow="Overview" title="Measuring a pulse from ordinary video">
        Every heartbeat pushes blood into the face and changes how much light the skin
        absorbs. The change is far too small to see, but a camera can measure it. This
        dashboard reports how accurately two methods recover a heart rate that way — a
        deep model called <span className="text-phase">PHASE-Net</span> and a classical
        algorithm called <span className="text-pos">POS</span> — and, just as
        importantly, where they fail.
      </PageHead>

      <Grid cols={4}>
        <Stat
          label="Accuracy, clean data"
          value={fmt(h.held_out_mae_vs_reference)}
          unit="BPM off"
          tone="phase"
          compare={{ value: fmt(h.paper_claim), label: "claimed by the paper" }}
          hint={`Average error on ${h.n_held_out_subjects} subjects the model had never seen.`}
        />
        <Stat
          label="Against the oximeter"
          value={fmt(h.held_out_mae_vs_device)}
          unit="BPM off"
          tone="good"
          hint="Checked against a medical device's own reading, which our code cannot influence."
        />
        <Stat
          label="This webcam, at rest"
          value={fmt(h.webcam_noise_floor, 1)}
          unit="BPM off"
          tone="warn"
          hint="Sitting perfectly still, nothing moving. This is the number that limits everything else."
        />
        <Stat
          label="Measurements taken"
          value={h.total_windows?.toLocaleString() ?? "—"}
          unit="windows"
          hint="Ten seconds each. Heart rate changes within a minute, so one value per video is not enough."
        />
      </Grid>

      <Panel
        title="So how far off is it?"
        hint="The same measurement in four situations. Shorter is better; the shaded band is within 3 BPM of the truth."
      >
        <ErrorScale rows={scale} />
        <ChartNote>
          On controlled recordings the model is close to what its authors report. Point a
          laptop webcam at someone sitting perfectly still and the bar runs off the end of
          that band — before anyone has moved. That gap, not movement, is the thing worth
          fixing.
        </ChartNote>
      </Panel>

      <Grid cols={3}>
        <Panel title="It works on good data" hint="Clinical validation">
          <p className="text-sm leading-relaxed text-ink-soft">
            Measured against a pulse oximeter on the UBFC-rPPG dataset, the model is off by{" "}
            <span className="num text-phase">{fmt(h.held_out_mae_vs_reference)} BPM</span>{" "}
            on average.
          </p>
          <p className="mt-2 text-sm leading-relaxed text-ink-soft">
            Getting there needed the evaluation fixed first: measuring one heart rate per
            video averages away a quantity that changes within the minute.
          </p>
          <Link
            href="/validation"
            className="mt-3 inline-block text-xs text-phase underline underline-offset-4"
          >
            See the agreement plots →
          </Link>
        </Panel>

        <Panel title="Its edge is reliability" hint="Model comparison">
          <div className="mb-3 flex items-end gap-4">
            <div>
              <div className="text-2xs uppercase tracking-wider text-ink-faint">
                Worst case
              </div>
              <div className="num mt-0.5 text-2xl font-semibold text-phase">
                {fmt(worstPhase, 1)}
              </div>
              <div className="text-2xs text-ink-dim">PHASE-Net</div>
            </div>
            <div className="pb-1 text-ink-dim">vs</div>
            <div>
              <div className="num mt-0.5 text-2xl font-semibold text-pos">
                {fmt(worstPos, 1)}
              </div>
              <div className="text-2xs text-ink-dim">POS</div>
            </div>
          </div>
          <p className="text-sm leading-relaxed text-ink-soft">
            On average the two tie. What separates them is how badly they fail on their
            worst subject — and for monitoring a patient, never being badly wrong matters
            more than a better average.
          </p>
          <Link
            href="/comparison"
            className="mt-3 inline-block text-xs text-phase underline underline-offset-4"
          >
            Compare the methods →
          </Link>
        </Panel>

        <Panel title="Three ideas, all wrong" hint="Motion protocol">
          <div className="mb-3 space-y-1.5">
            {[
              "The face leaves the crop",
              "Talking changes the face",
              "One method is more robust",
            ].map((t) => (
              <div key={t} className="flex items-center gap-2 text-xs">
                <Chip tone="bad">rejected</Chip>
                <span className="text-ink-soft">{t}</span>
              </div>
            ))}
          </div>
          <p className="text-sm leading-relaxed text-ink-soft">
            Each was tested and each failed. What the experiments found instead was the
            noise floor above — the error is already large before anyone moves.
          </p>
          <Link
            href="/motion"
            className="mt-3 inline-block text-xs text-phase underline underline-offset-4"
          >
            Read the experiments →
          </Link>
        </Panel>
      </Grid>

      <Grid cols={2}>
        <Callout tone="warn" title="What these numbers do not show">
          <ul className="list-disc space-y-1.5 pl-4 text-xs leading-relaxed">
            {d.limitations.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </Callout>

        <Panel title="Two references, because neither alone is trustworthy">
          <div className="space-y-3 text-sm leading-relaxed text-ink-soft">
            <div>
              <div className="mb-1 flex items-center gap-2">
                <Chip tone="phase">vs reference waveform</Chip>
                <span className="text-2xs text-ink-dim">what papers report</span>
              </div>
              <p className="text-xs">
                The same processing runs on the prediction and on the ground-truth pulse.
                Comparable to published figures — but shared processing can hide shared
                mistakes. Scoring a method against itself once made a result look{" "}
                <span className="text-warn">eighteen times better</span> than it was.
              </p>
            </div>
            <div>
              <div className="mb-1 flex items-center gap-2">
                <Chip tone="good">vs the oximeter</Chip>
                <span className="text-2xs text-ink-dim">stricter</span>
              </div>
              <p className="text-xs">
                The device reports its own heart rate, untouched by our code, so it cannot
                flatter us. It is not perfect either: on two subjects it is wrong by more
                than 20 BPM, which two independent analyses of its own waveform confirm.
              </p>
            </div>
          </div>
          <ChartNote>
            Both are reported on every page. Where they disagree, the disagreement is the
            interesting part.
          </ChartNote>
        </Panel>
      </Grid>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line pt-4 text-2xs text-ink-dim">
        <span>Generated {new Date(d.generated_utc).toUTCString()}</span>
        <span>·</span>
        <span>schema {d.schema_version}</span>
        <span>·</span>
        <Link href="/reproducibility" className="text-ink-faint underline underline-offset-2">
          weights hash, git commit and library versions
        </Link>
      </div>
    </>
  );
}
