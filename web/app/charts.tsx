export interface ScaleRow {
  name: string;
  value: number;
  colour: string;
  note: string;
}

/**
 * The whole project on one axis: what the paper claims, what we measured on clean data,
 * and how far off that a laptop webcam is.
 *
 * Deliberately plain CSS rather than a charting library. Four horizontal bars need no
 * chart engine, and Recharts was mis-scaling the bars against a category axis inside a
 * narrow panel. Bars laid out as divs are exact by construction.
 *
 * The scale is LINEAR on purpose. A logarithmic axis would compress the very gap this
 * chart exists to show - the webcam bar has to visibly dwarf the others, because that
 * twenty-fold difference is the finding.
 */
export function ErrorScale({ rows }: { rows: ScaleRow[] }) {
  const max = Math.ceil(Math.max(...rows.map((r) => r.value)));
  const ticks = Array.from({ length: max + 1 }, (_, i) => i).filter(
    (t) => max <= 6 || t % 2 === 0
  );
  const clean = rows.find((r) => r.name.startsWith("Measured"))?.value ?? 0;
  const webcam = rows[rows.length - 1]?.value ?? 0;
  const ratio = clean > 0 ? webcam / clean : 0;
  const pct = (v: number) => `${(v / max) * 100}%`;

  return (
    <div>
      <div className="flex gap-3">
        {/* labels */}
        <div className="w-40 shrink-0 space-y-2 pt-0.5 sm:w-52">
          {rows.map((r) => (
            <div
              key={r.name}
              className="flex h-7 items-center justify-end text-right text-xs text-ink-soft"
            >
              {r.name}
            </div>
          ))}
        </div>

        {/* plot */}
        <div className="relative min-w-0 flex-1">
          {/* the band inside which a reading is close enough to be useful */}
          <div
            className="absolute inset-y-0 left-0 rounded-sm bg-good/[0.07]"
            style={{ width: pct(3) }}
            aria-hidden
          />
          {/* gridlines */}
          {ticks.map((t) => (
            <div
              key={t}
              className="absolute inset-y-0 w-px bg-line"
              style={{ left: pct(t) }}
              aria-hidden
            />
          ))}

          <div className="relative space-y-2">
            {rows.map((r) => (
              <div key={r.name} className="flex h-7 items-center gap-2">
                <div
                  className="h-5 rounded-sm transition-all"
                  style={{
                    width: `max(3px, ${pct(r.value)})`,
                    backgroundColor: r.colour,
                  }}
                  title={r.note}
                />
                <span className="num shrink-0 text-xs text-ink">
                  {r.value.toFixed(2)}
                </span>
              </div>
            ))}
          </div>

          {/* axis */}
          <div className="relative mt-2 h-8 border-t border-line">
            {ticks.map((t) => (
              <span
                key={t}
                className="num absolute -translate-x-1/2 pt-1 text-2xs text-ink-faint"
                style={{ left: pct(t) }}
              >
                {t}
              </span>
            ))}
            <div className="absolute inset-x-0 top-5 text-center text-2xs text-ink-dim">
              BPM away from the true heart rate
            </div>
          </div>
        </div>
      </div>

      {ratio > 1 && (
        <div className="mt-3 flex items-center gap-2.5 rounded-md border border-warn/25 bg-warn/[0.06] px-3 py-2">
          <span className="num text-xl font-semibold leading-none text-warn">
            {ratio.toFixed(0)}×
          </span>
          <span className="text-xs leading-snug text-ink-soft">
            worse on a laptop webcam than the same model achieves on controlled
            recordings — with nobody moving.
          </span>
        </div>
      )}
    </div>
  );
}
