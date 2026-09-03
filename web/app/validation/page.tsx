import { getDashboard, fmt, PALETTE } from "@/lib/data";
import type { Num, Subject } from "@/lib/types";
import {
  BarCell,
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
import {
  AgreementPlot,
  ErrorVsQuality,
  PerSubjectError,
  PredictedVsReference,
  type AgreementPoint,
  type ErrorBar,
  type SnrPoint,
} from "./charts";

/** The exporter writes null where Python had NaN; treat both as missing. */
function num(v: Num | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** subject5, subject44 ... sort by the number, not the string. */
function idNumber(id: string): number {
  const digits = id.replace(/\D/g, "");
  return digits.length > 0 ? Number(digits) : 0;
}

/** Render a configuration value from the JSON without pretending to know its type. */
function show(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (Array.isArray(v)) {
    return v.every((x) => typeof x === "number") ? v.join(" × ") : v.map(show).join(", ");
  }
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function defItems(source: Record<string, unknown>, skip: string[] = []) {
  return Object.entries(source)
    .filter(([k]) => !skip.includes(k))
    .map(([k, v]) => [k.replace(/_/g, " "), show(v)] as [string, string]);
}

/** Which dots are which, in the panel header where the eye looks first. */
function SplitLegend({ held, trained }: { held: number; trained: number }) {
  return (
    <div className="flex items-center gap-3">
      <span className="flex items-center gap-1.5">
        <Dot colour={PALETTE.phasenet} /> held out ({held})
      </span>
      <span className="flex items-center gap-1.5">
        <Dot colour={PALETTE.axis} /> trained on ({trained})
      </span>
    </div>
  );
}

export default async function Validation() {
  const d = await getDashboard();
  const val = d.validation;

  if (!val) {
    return (
      <>
        <PageHead eyebrow="Validation" title="Does it agree with a medical device?">
          The validation block is missing from this export.
        </PageHead>
        <Panel title="No data">
          <p className="text-sm text-ink-soft">
            Re-run the exporter after <span className="num">04_ubfc_eval.py</span> and{" "}
            <span className="num">08_windowed_eval.py</span> have written their results.
          </p>
        </Panel>
      </>
    );
  }

  const mvl = val.metrics_video_level;
  const mw = val.metrics_windowed;
  const ba = mvl?.bland_altman;

  const agreement: AgreementPoint[] = [];
  const snr: SnrPoint[] = [];
  for (const s of val.subjects) {
    const meanHr = num(s.mean_hr);
    const error = num(s.error);
    const reference = num(s.hr_reference);
    const predicted = num(s.hr_predicted);
    if (meanHr !== null && error !== null && reference !== null && predicted !== null) {
      agreement.push({ id: s.id, heldOut: s.held_out, meanHr, error, reference, predicted });
    }
    const snrDb = num(s.snr_db);
    const absError = num(s.abs_error);
    if (snrDb !== null && absError !== null) {
      snr.push({ id: s.id, heldOut: s.held_out, snrDb, absError });
    }
  }

  const bars: ErrorBar[] = val.subjects
    .map((s) => ({
      id: s.id,
      heldOut: s.held_out,
      absError: num(s.abs_error),
      windowedMae: num(s.windowed?.paired_mae),
    }))
    .filter((b): b is ErrorBar => b.absError !== null)
    .sort((a, b) => b.absError - a.absError);

  const rows: Subject[] = [...val.subjects].sort(
    (a, b) => Number(b.held_out) - Number(a.held_out) || idNumber(a.id) - idNumber(b.id)
  );

  const nAll = val.subjects.length;
  const nHeld = val.n_held_out;
  const nTrained = nAll - nHeld;
  const exactlyZero = bars.filter((b) => b.absError === 0).length;
  const worst = bars.find((b) => b.heldOut) ?? null;
  const errorMax = Math.max(1, ...bars.map((b) => b.absError));

  return (
    <>
      <PageHead eyebrow="Validation" title="Does it agree with a medical device?">
        The camera&apos;s heart rate was compared with a fingertip pulse oximeter on {nAll}{" "}
        recordings, and only {nHeld} of them count: the released model was trained on the
        first 72 per cent of the subject list, so on the other {nTrained} it would be
        measuring memory rather than skill.
      </PageHead>

      <Grid cols={4}>
        <Stat
          label="Off by, vs the pulse trace"
          value={fmt(mw?.held_out?.paired_mae)}
          unit="BPM"
          tone="phase"
          meter={num(mw?.held_out?.paired_mae)}
          meterMax={9}
          hint={`${nHeld} unseen people, ten-second windows. The oximeter's recorded pulse, put through the same processing as the camera signal.`}
        />
        <Stat
          label="Off by, vs the device readout"
          value={fmt(mw?.held_out?.device_mae)}
          unit="BPM"
          tone="good"
          meter={num(mw?.held_out?.device_mae)}
          meterMax={9}
          hint={`The number the device itself displays, which our code cannot influence — the stricter of the two. Across all ${nAll} it is ${fmt(
            mw?.all?.device_mae
          )}, inflated by two recordings where the device's own readout is wrong by over 20 BPM.`}
        />
        <Stat
          label="Agreement"
          value={fmt(mvl?.pearson_r, 3)}
          unit="Pearson r"
          compare={{ value: "1.00", label: "would be perfect" }}
          hint="Whether the two rise and fall together across subjects. Blind to a constant offset, which the plot below is not."
        />
        <Stat
          label="Within 5 BPM"
          value={fmt(mvl?.within_5bpm_pct, 0)}
          unit={`% of ${nAll}`}
          tone="warn"
          meter={num(mvl?.within_5bpm_pct)}
          meterMax={100}
          hint={`${fmt(mvl?.within_3bpm_pct, 0)} % land within 3 BPM. One reading per recording, all subjects.`}
        />
      </Grid>

      <Panel
        title="Agreement plot"
        hint={`Bland-Altman: the standard way clinicians check whether a new measurement agrees with an established one. All ${nAll} recordings, ${nTrained} of which the model has seen.`}
        right={<SplitLegend held={nHeld} trained={nTrained} />}
      >
        <AgreementPlot
          points={agreement}
          bias={num(ba?.bias_bpm)}
          loaLower={num(ba?.loa_lower_bpm)}
          loaUpper={num(ba?.loa_upper_bpm)}
        />
        <ChartNote>
          Each dot is one person. The solid line is the average gap between camera and
          reference — <span className="num">{fmt(ba?.bias_bpm)} BPM</span>, so the camera does
          not systematically read high or low. Nineteen readings in twenty should fall between
          the dashed lines at <span className="num">{fmt(ba?.loa_lower_bpm, 1)}</span> and{" "}
          <span className="num">{fmt(ba?.loa_upper_bpm, 1)} BPM</span>, and that width, not the
          average, is what a monitoring claim would have to live with.
        </ChartNote>
      </Panel>

      <Grid cols={2}>
        <Panel
          title="Camera against reference"
          hint="One reading per recording"
          right={<SplitLegend held={nHeld} trained={nTrained} />}
        >
          <PredictedVsReference points={agreement} />
          <ChartNote>
            Dots on the dashed line are exact; above it the camera read fast, below it slow.
            The spread along the line is the range of resting heart rates in the group, not
            error.
          </ChartNote>
        </Panel>

        <Panel title="Who it fails on" hint="Worst first, whole recording">
          <PerSubjectError bars={bars} />
          <ChartNote>
            {exactlyZero} of {nAll} sit at zero, drawn as a hairline. That is a caveat, not
            applause: the frequency analysis resolves about 0.88 BPM, so zero means the same
            bin.{" "}
            {worst && (
              <>
                The worst unseen person is <span className="num">{worst.id}</span> at{" "}
                <span className="num">{fmt(worst.absError, 1)} BPM</span> — and{" "}
                <span className="num">{fmt(worst.windowedMae)}</span> once the same recording
                is scored in ten-second windows.
              </>
            )}
          </ChartNote>
        </Panel>
      </Grid>

      <Panel
        title="Can it tell when it is struggling?"
        hint="Signal quality against error, one dot per person"
        right={<SplitLegend held={nHeld} trained={nTrained} />}
      >
        <ErrorVsQuality points={snr} />
        <ChartNote>
          Signal quality is how far the pulse stands above the noise, and it can be measured
          live with no reference to compare against. If a low score reliably meant a large
          error, the model could refuse to answer — a monitoring product has to be able to say
          &ldquo;I do not know&rdquo; rather than guess. It cannot yet: everything above 10 dB
          is exact, but below that an exact reading and an 8 BPM miss sit side by side.
        </ChartNote>
      </Panel>

      <Panel
        title="Every recording"
        hint={`All ${nAll}, unseen subjects first. Nothing averaged away.`}
        right={`${mw?.total_windows?.toLocaleString() ?? "—"} windows in total`}
      >
        <Table
          head={["Person", "Split", "Reference", "Camera", "Off by", "Signal", "Windowed"]}
          align={["l", "l", "r", "r", "r", "r", "r"]}
        >
          {rows.map((s) => (
            <Row key={s.id} highlight={s.held_out}>
              <Cell strong>{s.id}</Cell>
              <Cell>
                {s.held_out ? <Chip tone="phase">held out</Chip> : <Chip>trained on</Chip>}
              </Cell>
              <Cell num>{fmt(s.hr_reference, 1)}</Cell>
              <Cell num>{fmt(s.hr_predicted, 1)}</Cell>
              <BarCell
                value={num(s.abs_error)}
                max={errorMax}
                colour={s.held_out ? PALETTE.phasenet : PALETTE.axis}
              />
              <Cell num muted>
                {fmt(s.snr_db, 1)}
              </Cell>
              <Cell num>{fmt(s.windowed?.paired_mae)}</Cell>
            </Row>
          ))}
        </Table>
        <ChartNote>
          Reference, camera and off-by are in BPM; signal is in dB. The last column re-scores
          the same recording in ten-second windows — heart rate moves within a minute, so one
          number per video averages away the thing being measured.
        </ChartNote>
      </Panel>

      <Panel
        title="How the video was processed"
        hint="Every step between the raw frames and the numbers above"
      >
        <DefList
          items={[...defItems(val.preprocessing ?? {}), ...defItems(val.hr_extraction ?? {})]}
        />
        <ChartNote>
          One line there decides whether any of this works: pixel values stay in the 0–255
          range. The reference implementation never divides by 255, so normalising raises no
          error and silently returns a plausible-looking waveform that is noise.
        </ChartNote>
      </Panel>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line pt-4 text-2xs text-ink-dim">
        <span className="num">04_ubfc_eval.py</span>
        <span>·</span>
        <span className="num">08_windowed_eval.py</span>
        <span>·</span>
        <span>exported {new Date(d.generated_utc).toUTCString()}</span>
        <span>·</span>
        <span>schema {d.schema_version}</span>
      </div>
    </>
  );
}
