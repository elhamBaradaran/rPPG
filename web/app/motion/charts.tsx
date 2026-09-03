"use client";

// Recharts needs the browser, so the charts live here and the page stays a server
// component. Nothing is computed in this file - it receives plain arrays and draws them.
// Imports come from "@/lib/format" only: "@/lib/data" touches node:fs and cannot be
// bundled for the client.

import {
  Bar,
  BarChart,
  CartesianGrid,
  Label,
  LabelList,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS, PALETTE, TOOLTIP, methodColour } from "@/lib/format";

/** One condition: how much movement was measured, and how far each method strayed. */
export interface DriftRow {
  /** Friendly condition name, used as the x tick and the dot label. */
  label: string;
  /** Measured movement during the condition, above its own still baseline. */
  dose: number;
  /** Highest drift of any method here - carries the dot labels above every line. */
  top: number;
  [method: string]: string | number | null;
}

export interface FloorLine {
  name: string;
  value: number;
}

const LABEL = { fill: PALETTE.axis, fontSize: 11 } as const;
const AXIS_TITLE = { fill: PALETTE.axis, fontSize: 11 } as const;
const LEGEND = { fontSize: 11, paddingBottom: 4 } as const;

/**
 * Drift against measured movement. The x axis is the measured quantity itself rather
 * than the condition name, so the horizontal spacing carries the finding: the last two
 * conditions sit far apart on the axis and at the same height.
 */
export function DoseResponse({
  rows,
  methods,
  plateauFrom,
}: {
  rows: DriftRow[];
  methods: string[];
  plateauFrom?: number;
}) {
  const maxDose = rows.reduce((a, r) => Math.max(a, r.dose), 0);

  return (
    <div className="h-[320px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 22, right: 30, bottom: 26, left: 4 }}>
          <CartesianGrid stroke={PALETTE.grid} strokeDasharray="3 3" />
          {plateauFrom !== undefined && (
            <ReferenceArea
              x1={plateauFrom}
              x2={maxDose + 0.35}
              fill={PALETTE.warn}
              fillOpacity={0.05}
              stroke="none"
            >
              <Label
                value="no further effect"
                position="insideTop"
                offset={8}
                fill={PALETTE.axis}
                fontSize={10}
              />
            </ReferenceArea>
          )}
          <XAxis
            dataKey="dose"
            type="number"
            domain={[-0.1, maxDose + 0.35]}
            tickFormatter={(v: number) => v.toFixed(1)}
            tick={AXIS.tick}
            stroke={AXIS.stroke}
          >
            <Label
              value="movement measured during the condition"
              position="insideBottom"
              offset={-16}
              {...AXIS_TITLE}
            />
          </XAxis>
          <YAxis
            tick={AXIS.tick}
            stroke={AXIS.stroke}
            domain={[0, "dataMax + 4"]}
            tickFormatter={(v: number) => v.toFixed(0)}
            width={44}
          >
            <Label
              value="BPM off"
              angle={-90}
              position="insideLeft"
              style={{ textAnchor: "middle" }}
              {...AXIS_TITLE}
            />
          </YAxis>
          <Tooltip
            {...TOOLTIP}
            formatter={(v: number, n: string) => [`${v.toFixed(1)} BPM off`, n]}
            labelFormatter={(v: number) => `movement ${Number(v).toFixed(2)}`}
          />
          <Legend
            verticalAlign="top"
            align="right"
            height={26}
            iconType="circle"
            iconSize={8}
            wrapperStyle={LEGEND}
          />
          {methods.map((name) => (
            <Line
              key={name}
              type="linear"
              dataKey={name}
              name={name}
              stroke={methodColour(name)}
              strokeWidth={2}
              dot={{ r: 4, strokeWidth: 0, fill: methodColour(name) }}
              activeDot={{ r: 6 }}
              connectNulls
              isAnimationActive={false}
            />
          ))}
          {/* Invisible series pinned to the tallest point of each condition, so the
              condition names sit clear of both lines instead of colliding with
              whichever happens to be on top. */}
          <Line
            dataKey="top"
            stroke="transparent"
            dot={false}
            activeDot={false}
            legendType="none"
            tooltipType="none"
            isAnimationActive={false}
          >
            <LabelList dataKey="label" position="top" offset={12} style={LABEL} />
          </Line>
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * The same drift values as bars, ordered by measured movement, with each method's
 * still-only error drawn as a dashed line. A bar level with its own dashed line is a
 * condition that added nothing over sitting there.
 */
export function DriftBars({
  rows,
  methods,
  floors,
}: {
  rows: DriftRow[];
  methods: string[];
  floors: FloorLine[];
}) {
  return (
    <div className="h-[320px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 8, right: 104, bottom: 24, left: 4 }}>
          <CartesianGrid stroke={PALETTE.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" tick={AXIS.tick} stroke={AXIS.stroke}>
            <Label
              value="least movement on the left, most on the right"
              position="insideBottom"
              offset={-14}
              {...AXIS_TITLE}
            />
          </XAxis>
          <YAxis
            tick={AXIS.tick}
            stroke={AXIS.stroke}
            domain={[0, "dataMax + 4"]}
            tickFormatter={(v: number) => v.toFixed(0)}
            width={44}
          >
            <Label
              value="BPM off"
              angle={-90}
              position="insideLeft"
              style={{ textAnchor: "middle" }}
              {...AXIS_TITLE}
            />
          </YAxis>
          <Tooltip
            {...TOOLTIP}
            cursor={{ fill: "rgba(255,255,255,0.03)" }}
            formatter={(v: number, n: string) => [`${v.toFixed(1)} BPM off`, n]}
          />
          <Legend
            verticalAlign="top"
            align="right"
            height={26}
            iconType="circle"
            iconSize={8}
            wrapperStyle={LEGEND}
          />
          {methods.map((name) => (
            <Bar
              key={name}
              dataKey={name}
              name={name}
              fill={methodColour(name)}
              radius={[3, 3, 0, 0]}
              isAnimationActive={false}
            >
              <LabelList
                dataKey={name}
                position="top"
                formatter={(v: number) => (typeof v === "number" ? v.toFixed(1) : "")}
                style={{ ...LABEL, fontSize: 10, fontFamily: "ui-monospace" }}
              />
            </Bar>
          ))}
          {floors.map((f) => (
            <ReferenceLine
              key={f.name}
              y={f.value}
              stroke={methodColour(f.name)}
              strokeDasharray="5 4"
              strokeOpacity={0.8}
              ifOverflow="extendDomain"
            >
              <Label
                value={`${f.name} at rest`}
                position="right"
                fill={methodColour(f.name)}
                fontSize={10}
              />
            </ReferenceLine>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
