"use client";

// Recharts needs the browser, so every chart on this route lives here and the page stays a
// server component. This file must never import "@/lib/data" - that module reads the JSON
// off disk and would pull node:fs into the client bundle. Colours arrive already chosen by
// methodColour(), so a method keeps the same colour on every page and nothing is invented
// here. Axes, grid and hover card come from the shared AXIS / PALETTE / TOOLTIP constants,
// so these charts are indistinguishable in style from every other page's.

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS, PALETTE, TOOLTIP } from "@/lib/format";

const CURSOR = { fill: "rgba(255,255,255,0.03)" };
const LEGEND = { fontSize: 11, color: PALETTE.axis, paddingTop: 8 };
const VALUE_LABEL = {
  fill: PALETTE.axis,
  fontSize: 11,
  fontFamily: "ui-monospace, monospace",
};

export interface Series {
  /** Must match the keys used in GroupedRow.values. */
  key: string;
  /** Shown in the legend and the hover card. */
  name: string;
  colour: string;
}

export interface GroupedRow {
  label: string;
  /** One value per series key; null where the exporter wrote null for a NaN. */
  values: Record<string, number | null>;
}

/**
 * Categories along the x axis, one bar per method inside each category, so the reader
 * compares methods within a category rather than across the width of the chart.
 *
 * `dim` fades whole categories without removing them - used to push the nine people the
 * model was trained on into the background, so the six it has never seen read first.
 */
export function GroupedBars({
  rows,
  series,
  height = 300,
  angledTicks = false,
  labels = false,
  dim,
}: {
  rows: GroupedRow[];
  series: Series[];
  height?: number;
  /** For the fifteen-subject chart, where horizontal ticks would collide. */
  angledTicks?: boolean;
  /** Print the value above each bar. Only readable when there are few bars. */
  labels?: boolean;
  /** Category labels to render faded. */
  dim?: Set<string>;
}) {
  const data = rows.map((r) => ({ label: r.label, ...r.values }));

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <BarChart
          data={data}
          margin={{ top: 16, right: 8, left: 0, bottom: 0 }}
          barCategoryGap={angledTicks ? "14%" : "30%"}
        >
          <CartesianGrid stroke={PALETTE.grid} vertical={false} />
          <XAxis
            dataKey="label"
            interval={0}
            tick={AXIS.tick}
            tickLine={false}
            stroke={AXIS.stroke}
            angle={angledTicks ? -45 : 0}
            textAnchor={angledTicks ? "end" : "middle"}
            height={angledTicks ? 52 : 30}
          />
          <YAxis tick={AXIS.tick} tickLine={false} stroke={AXIS.stroke} width={40} />
          <Tooltip
            {...TOOLTIP}
            cursor={CURSOR}
            formatter={(v: number) => `${v.toFixed(2)} BPM`}
          />
          <Legend wrapperStyle={LEGEND} iconType="square" iconSize={9} />
          {series.map((s) => (
            <Bar
              key={s.key}
              dataKey={s.key}
              name={s.name}
              fill={s.colour}
              radius={[2, 2, 0, 0]}
              maxBarSize={56}
            >
              {dim
                ? data.map((r, i) => (
                    <Cell
                      key={i}
                      fill={s.colour}
                      fillOpacity={dim.has(r.label) ? 0.28 : 1}
                    />
                  ))
                : null}
              {labels ? (
                <LabelList
                  dataKey={s.key}
                  position="top"
                  formatter={(v: number) => v.toFixed(2)}
                  style={VALUE_LABEL}
                />
              ) : null}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export interface RankRow {
  name: string;
  value: number | null;
  colour: string;
  /** Extra line shown in the hover card. */
  note: string;
}

/**
 * One bar per method, laid out horizontally so the names stay readable and the length of
 * the bar is the whole message. Used for the worst-case chart, where the point is that one
 * bar is ten times the length of another.
 */
export function RankBars({
  rows,
  unit = "BPM",
  digits = 1,
}: {
  rows: RankRow[];
  unit?: string;
  digits?: number;
}) {
  return (
    <div style={{ width: "100%", height: Math.max(160, rows.length * 62) }}>
      <ResponsiveContainer>
        <BarChart
          data={rows}
          layout="vertical"
          margin={{ top: 4, right: 88, bottom: 4, left: 4 }}
        >
          <CartesianGrid stroke={PALETTE.grid} horizontal={false} />
          <XAxis
            type="number"
            domain={[0, "dataMax"]}
            tick={AXIS.tick}
            tickLine={false}
            stroke={AXIS.stroke}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={150}
            tick={{ ...AXIS.tick, fontSize: 12 }}
            tickLine={false}
            stroke={AXIS.stroke}
          />
          <Tooltip
            {...TOOLTIP}
            cursor={CURSOR}
            formatter={(v: number, _n, p) => [
              `${v.toFixed(2)} ${unit}`,
              (p?.payload as RankRow)?.note ?? "",
            ]}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={26}>
            {rows.map((r, i) => (
              <Cell key={i} fill={r.colour} />
            ))}
            <LabelList
              dataKey="value"
              position="right"
              formatter={(v: number) => `${v.toFixed(digits)} ${unit}`}
              style={VALUE_LABEL}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
