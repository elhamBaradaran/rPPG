import type { ReactNode } from "react";

/* ------------------------------------------------------------------ layout */

export function Panel({
  title,
  hint,
  right,
  children,
  className = "",
  pad = true,
}: {
  title?: string;
  hint?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return (
    <section
      className={`rounded-lg border border-line bg-bg-panel shadow-panel ${className}`}
    >
      {title && (
        <header className="flex items-start justify-between gap-3 border-b border-line-soft px-4 py-3">
          <div className="min-w-0">
            <h3>{title}</h3>
            {hint && <p className="mt-0.5 text-2xs leading-snug text-ink-faint">{hint}</p>}
          </div>
          {right && <div className="shrink-0 text-2xs text-ink-faint">{right}</div>}
        </header>
      )}
      <div className={pad ? "p-4" : ""}>{children}</div>
    </section>
  );
}

export function Grid({
  cols = 2,
  children,
  className = "",
}: {
  cols?: 2 | 3 | 4;
  children: ReactNode;
  className?: string;
}) {
  const c = { 2: "lg:grid-cols-2", 3: "lg:grid-cols-3", 4: "lg:grid-cols-2 xl:grid-cols-4" }[cols];
  return <div className={`grid gap-4 ${c} ${className}`}>{children}</div>;
}

export function PageHead({
  eyebrow,
  title,
  children,
}: {
  eyebrow?: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="border-b border-line pb-5">
      {eyebrow && (
        <div className="mb-1 text-2xs font-medium uppercase tracking-widest text-phase">
          {eyebrow}
        </div>
      )}
      <h1>{title}</h1>
      {children && (
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-ink-soft">{children}</p>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------- stats */

/**
 * A headline number. `meter` draws a bar showing where the value sits between 0 and
 * `meterMax`, which turns an abstract figure into something comparable at a glance.
 */
export function Stat({
  label,
  value,
  unit,
  hint,
  tone = "neutral",
  meter,
  meterMax,
  compare,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  hint?: string;
  tone?: "neutral" | "phase" | "pos" | "good" | "warn" | "bad";
  meter?: number | null;
  meterMax?: number;
  compare?: { label: string; value: string };
}) {
  const accent = {
    neutral: "text-ink",
    phase: "text-phase",
    pos: "text-pos",
    good: "text-good",
    warn: "text-warn",
    bad: "text-bad",
  }[tone];
  const bar = {
    neutral: "bg-ink-dim",
    phase: "bg-phase",
    pos: "bg-pos",
    good: "bg-good",
    warn: "bg-warn",
    bad: "bg-bad",
  }[tone];
  const pct =
    meter !== undefined && meter !== null && meterMax
      ? Math.max(2, Math.min(100, (meter / meterMax) * 100))
      : null;

  return (
    <div className="rounded-lg border border-line bg-bg-panel p-4 shadow-panel">
      <div className="text-2xs uppercase tracking-wider text-ink-faint">{label}</div>
      <div className="mt-1.5 flex items-baseline gap-1.5">
        <span className={`num text-3xl font-semibold leading-none ${accent}`}>{value}</span>
        {unit && <span className="text-xs text-ink-faint">{unit}</span>}
      </div>
      {pct !== null && (
        <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-line">
          <div className={`h-full rounded-full ${bar}`} style={{ width: `${pct}%` }} />
        </div>
      )}
      {compare && (
        <div className="mt-2 flex items-center gap-1.5 text-2xs text-ink-faint">
          <span className="num text-ink-soft">{compare.value}</span>
          <span>{compare.label}</span>
        </div>
      )}
      {hint && <p className="mt-2 text-2xs leading-snug text-ink-faint">{hint}</p>}
    </div>
  );
}

/* ------------------------------------------------------------------- chips */

export function Chip({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "phase" | "pos" | "good" | "warn" | "bad" | "info";
}) {
  const cls = {
    neutral: "border-line-bright bg-bg-raised text-ink-faint",
    phase: "border-phase/30 bg-phase/10 text-phase",
    pos: "border-pos/30 bg-pos/10 text-pos",
    good: "border-good/30 bg-good/10 text-good",
    warn: "border-warn/30 bg-warn/10 text-warn",
    bad: "border-bad/30 bg-bad/10 text-bad",
    info: "border-info/30 bg-info/10 text-info",
  }[tone];
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-0.5 text-2xs font-medium ${cls}`}
    >
      {children}
    </span>
  );
}

/** A coloured square that ties a label to a series colour. */
export function Dot({ colour }: { colour: string }) {
  return (
    <span
      className="inline-block h-2 w-2 shrink-0 rounded-sm"
      style={{ backgroundColor: colour }}
    />
  );
}

/* ------------------------------------------------------------------ callout */

export function Callout({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "warn" | "bad" | "good";
  title?: string;
  children: ReactNode;
}) {
  const cls = {
    info: "border-info/25 bg-info/[0.06]",
    warn: "border-warn/25 bg-warn/[0.06]",
    bad: "border-bad/25 bg-bad/[0.06]",
    good: "border-good/25 bg-good/[0.06]",
  }[tone];
  const titleCls = { info: "text-info", warn: "text-warn", bad: "text-bad", good: "text-good" }[
    tone
  ];
  return (
    <div className={`rounded-lg border p-4 ${cls}`}>
      {title && <h3 className={`mb-2 ${titleCls}`}>{title}</h3>}
      <div className="space-y-2 text-sm leading-relaxed text-ink-soft">{children}</div>
    </div>
  );
}

/* ------------------------------------------------------------------- table */

export function Table({
  head,
  children,
  align,
}: {
  head: ReactNode[];
  children: ReactNode;
  align?: ("l" | "r")[];
}) {
  return (
    <div className="scroll-slim -mx-4 overflow-x-auto px-4">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-2xs uppercase tracking-wider text-ink-faint">
            {head.map((h, i) => (
              <th
                key={i}
                className={`whitespace-nowrap px-2.5 py-2 font-medium ${
                  align?.[i] === "r" ? "text-right" : "text-left"
                }`}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Row({
  children,
  highlight = false,
}: {
  children: ReactNode;
  highlight?: boolean;
}) {
  return (
    <tr
      className={`border-b border-line-soft last:border-0 ${
        highlight ? "bg-phase/[0.04]" : ""
      } hover:bg-bg-raised/60`}
    >
      {children}
    </tr>
  );
}

export function Cell({
  children,
  num = false,
  strong = false,
  muted = false,
}: {
  children: ReactNode;
  num?: boolean;
  strong?: boolean;
  muted?: boolean;
}) {
  return (
    <td
      className={`whitespace-nowrap px-2.5 py-1.5 ${num ? "num text-right" : ""} ${
        strong ? "font-semibold text-ink" : ""
      } ${muted ? "text-ink-faint" : ""}`}
    >
      {children}
    </td>
  );
}

/**
 * A number with a bar behind it. Lets a column of values be compared by eye without
 * reading any of them - the point of a table in a dashboard rather than a report.
 */
export function BarCell({
  value,
  max,
  colour,
  digits = 2,
}: {
  value: number | null | undefined;
  max: number;
  colour: string;
  digits?: number;
}) {
  const ok = typeof value === "number" && !Number.isNaN(value);
  const pct = ok ? Math.max(1.5, Math.min(100, (value! / max) * 100)) : 0;
  return (
    <td className="px-2.5 py-1.5">
      <div className="flex items-center justify-end gap-2">
        <div className="hidden h-1.5 w-16 overflow-hidden rounded-full bg-line sm:block">
          <div
            className="h-full rounded-full"
            style={{ width: `${pct}%`, backgroundColor: colour }}
          />
        </div>
        <span className="num w-12 text-right text-ink">
          {ok ? value!.toFixed(digits) : "—"}
        </span>
      </div>
    </td>
  );
}

/* --------------------------------------------------------------- key/value */

export function DefList({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
      {items.map(([k, v]) => (
        <div key={k} className="min-w-0">
          <dt className="text-2xs uppercase tracking-wider text-ink-faint">{k}</dt>
          <dd className="num mt-0.5 break-all text-xs text-ink-soft">{v ?? "—"}</dd>
        </div>
      ))}
    </dl>
  );
}

/** One-line caption under a chart explaining how to read it. */
export function ChartNote({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 border-t border-line-soft pt-2.5 text-2xs leading-relaxed text-ink-faint">
      {children}
    </p>
  );
}
