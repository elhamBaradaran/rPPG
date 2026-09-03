"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS, PALETTE, TOOLTIP, fmt } from "@/lib/format";

// Recharts needs the browser, so every chart on this page lives here and the page itself
// stays a server component. Nothing is computed here beyond axis padding: the numbers
// arrive as plain arrays the page has already read from dashboard.json.
//
// Import rule: only "@/lib/format". "@/lib/data" touches node:fs and cannot be bundled.

/** Teal is the model everywhere on this dashboard; the trained-on subjects stay dim. */
const HELD_OUT = PALETTE.phasenet;
const TRAINED_ON = PALETTE.axis;

const LABEL = { fill: PALETTE.axis, fontSize: 11 } as const;
const GRID = { stroke: PALETTE.grid, strokeDasharray: "3 3" } as const;
const CURSOR_DOT = { stroke: PALETTE.grid, strokeDasharray: "3 3" } as const;

export interface AgreementPoint {
  id: string;
  heldOut: boolean;
  meanHr: number;
  error: number;
  reference: number;
  predicted: number;
}

export interface ErrorBar {
  id: string;
  heldOut: boolean;
  absError: number;
  windowedMae: number | null;
}

export interface SnrPoint {
  id: string;
  heldOut: boolean;
  snrDb: number;
  absError: number;
  /** Waveform-shape agreement, carried through so the tooltip can show it. */
  macc?: number | null;
}

/** Axis range with a little breathing room, tolerating an empty series. */
function padded(values: number[], frac = 0.08): [number, number] {
  if (values.length === 0) return [0, 1];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || Math.max(Math.abs(hi), 1);
  return [lo - span * frac, hi + span * frac];
}

/** Put the subject and its split in the tooltip header, where the axis value would be. */
function subjectLabel(_label: unknown, payload: unknown): string {
  const first = (payload as Array<{ payload?: { id?: string; heldOut?: boolean } }> | undefined)?.[0]
    ?.payload;
  if (!first?.id) return "";
  return `${first.id} · ${first.heldOut ? "held out" : "training split"}`;
}

const bpm = (v: number, name: string): [string, string] => [`${v.toFixed(2)} BPM`, name];

/**
 * Bland-Altman. Each dot is one person, plotted at the average of the two readings
 * against the gap between them, so a constant offset that correlation would score as
 * perfect shows up as a line sitting away from zero.
 */
export function AgreementPlot({
  points,
  bias,
  loaLower,
  loaUpper,
}: {
  points: AgreementPoint[];
  bias: number | null;
  loaLower: number | null;
  loaUpper: number | null;
}) {
  const lines = [bias, loaLower, loaUpper].filter((v): v is number => v !== null);
  const yDomain = padded([...points.map((p) => p.error), ...lines, 0], 0.2);
  const xDomain = padded(points.map((p) => p.meanHr));

  return (
    <ResponsiveContainer width="100%" height={340}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 26, left: 4 }}>
        <CartesianGrid {...GRID} />
        <XAxis
          type="number"
          dataKey="meanHr"
          name="average of both"
          domain={xDomain}
          tick={AXIS.tick}
          stroke={AXIS.stroke}
          tickFormatter={(v: number) => v.toFixed(0)}
          label={{
            value: "average heart rate of the two readings (BPM)",
            position: "insideBottom",
            offset: -16,
            ...LABEL,
          }}
        />
        <YAxis
          type="number"
          dataKey="error"
          name="camera minus reference"
          domain={yDomain}
          tick={AXIS.tick}
          stroke={AXIS.stroke}
          tickFormatter={(v: number) => v.toFixed(0)}
          label={{
            value: "camera − reference (BPM)",
            angle: -90,
            position: "insideLeft",
            offset: 14,
            ...LABEL,
          }}
        />
        <ReferenceLine y={0} stroke={PALETTE.grid} />
        {loaUpper !== null && (
          <ReferenceLine
            y={loaUpper}
            stroke={PALETTE.bad}
            strokeDasharray="6 4"
            label={{
              value: `+${fmt(loaUpper, 1)}`,
              position: "insideTopRight",
              fill: PALETTE.bad,
              fontSize: 11,
            }}
          />
        )}
        {loaLower !== null && (
          <ReferenceLine
            y={loaLower}
            stroke={PALETTE.bad}
            strokeDasharray="6 4"
            label={{
              value: `${fmt(loaLower, 1)}`,
              position: "insideBottomRight",
              fill: PALETTE.bad,
              fontSize: 11,
            }}
          />
        )}
        {bias !== null && (
          <ReferenceLine
            y={bias}
            stroke={PALETTE.reference}
            strokeWidth={1.5}
            label={{
              value: `average gap ${fmt(bias, 2)} BPM`,
              position: "insideTopLeft",
              fill: PALETTE.reference,
              fontSize: 11,
            }}
          />
        )}
        <Tooltip {...TOOLTIP} cursor={CURSOR_DOT} labelFormatter={subjectLabel} formatter={bpm} />
        <Scatter data={points.filter((p) => !p.heldOut)} fill={TRAINED_ON} />
        <Scatter data={points.filter((p) => p.heldOut)} fill={HELD_OUT} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/** The familiar view: camera reading against reference reading, with the y = x line. */
export function PredictedVsReference({ points }: { points: AgreementPoint[] }) {
  const domain = padded([
    ...points.map((p) => p.reference),
    ...points.map((p) => p.predicted),
  ]);

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ScatterChart margin={{ top: 8, right: 14, bottom: 26, left: 4 }}>
        <CartesianGrid {...GRID} />
        <XAxis
          type="number"
          dataKey="reference"
          name="reference"
          domain={domain}
          tick={AXIS.tick}
          stroke={AXIS.stroke}
          tickFormatter={(v: number) => v.toFixed(0)}
          label={{ value: "reference (BPM)", position: "insideBottom", offset: -16, ...LABEL }}
        />
        <YAxis
          type="number"
          dataKey="predicted"
          name="camera"
          domain={domain}
          tick={AXIS.tick}
          stroke={AXIS.stroke}
          tickFormatter={(v: number) => v.toFixed(0)}
          label={{
            value: "camera (BPM)",
            angle: -90,
            position: "insideLeft",
            offset: 14,
            ...LABEL,
          }}
        />
        <ReferenceLine
          segment={[
            { x: domain[0], y: domain[0] },
            { x: domain[1], y: domain[1] },
          ]}
          stroke={PALETTE.reference}
          strokeDasharray="4 4"
        />
        <Tooltip {...TOOLTIP} cursor={CURSOR_DOT} labelFormatter={subjectLabel} formatter={bpm} />
        <Scatter data={points.filter((p) => !p.heldOut)} fill={TRAINED_ON} />
        <Scatter data={points.filter((p) => p.heldOut)} fill={HELD_OUT} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/**
 * One bar per person, worst first, so the average cannot hide who it fails on. Exact
 * readings are drawn as a hairline rather than nothing, so an empty row still reads as a
 * measurement and not as missing data.
 */
export function PerSubjectError({ bars }: { bars: ErrorBar[] }) {
  return (
    <ResponsiveContainer width="100%" height={Math.max(300, bars.length * 20 + 44)}>
      <BarChart layout="vertical" data={bars} margin={{ top: 4, right: 16, bottom: 22, left: 0 }}>
        <CartesianGrid horizontal={false} {...GRID} />
        <XAxis
          type="number"
          tick={AXIS.tick}
          stroke={AXIS.stroke}
          label={{
            value: "how far off, whole recording (BPM)",
            position: "insideBottom",
            offset: -12,
            ...LABEL,
          }}
        />
        <YAxis type="category" dataKey="id" width={78} tick={AXIS.tick} stroke={AXIS.stroke} />
        <Tooltip
          {...TOOLTIP}
          cursor={{ fill: "rgba(255,255,255,0.03)" }}
          labelFormatter={subjectLabel}
          formatter={(v: number, _n, item) => [
            `${v.toFixed(2)} BPM`,
            `whole recording · in 10 s windows ${fmt(
              (item?.payload as ErrorBar | undefined)?.windowedMae
            )}`,
          ]}
        />
        <Bar dataKey="absError" radius={[0, 3, 3, 0]} barSize={11} minPointSize={2}>
          {bars.map((b) => (
            <Cell key={b.id} fill={b.heldOut ? HELD_OUT : TRAINED_ON} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Signal strength against error: is the model's own quality score worth trusting. */
export function ErrorVsQuality({ points }: { points: SnrPoint[] }) {
  const xDomain = padded(points.map((p) => p.snrDb));
  const yDomain = padded([...points.map((p) => p.absError), 0], 0.16);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 26, left: 4 }}>
        <CartesianGrid {...GRID} />
        <XAxis
          type="number"
          dataKey="snrDb"
          name="signal quality"
          domain={xDomain}
          tick={AXIS.tick}
          stroke={AXIS.stroke}
          tickFormatter={(v: number) => v.toFixed(0)}
          label={{
            value: "signal quality (dB) — higher is a cleaner pulse",
            position: "insideBottom",
            offset: -16,
            ...LABEL,
          }}
        />
        <YAxis
          type="number"
          dataKey="absError"
          name="error"
          domain={yDomain}
          tick={AXIS.tick}
          stroke={AXIS.stroke}
          tickFormatter={(v: number) => v.toFixed(0)}
          label={{
            value: "how far off (BPM)",
            angle: -90,
            position: "insideLeft",
            offset: 14,
            ...LABEL,
          }}
        />
        <ReferenceLine x={0} stroke={PALETTE.reference} strokeOpacity={0.45} strokeDasharray="4 4" />
        <Tooltip
          {...TOOLTIP}
          cursor={CURSOR_DOT}
          labelFormatter={subjectLabel}
          formatter={(v: number, name: string) => [
            name === "signal quality" ? `${v.toFixed(1)} dB` : `${v.toFixed(2)} BPM`,
            name,
          ]}
        />
        <Scatter data={points.filter((p) => !p.heldOut)} fill={TRAINED_ON} />
        <Scatter data={points.filter((p) => p.heldOut)} fill={HELD_OUT} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
