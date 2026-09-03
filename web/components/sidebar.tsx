"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV: { href: string; label: string; blurb: string }[] = [
  { href: "/", label: "Overview", blurb: "The result in one screen" },
  { href: "/validation", label: "Validation", blurb: "Against a pulse oximeter" },
  { href: "/comparison", label: "Comparison", blurb: "Deep model vs classical" },
  { href: "/motion", label: "Motion", blurb: "What breaks it, and what does not" },
  { href: "/reproducibility", label: "Provenance", blurb: "How the numbers were made" },
];

export function Sidebar() {
  const path = usePathname();
  return (
    <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-line bg-bg-raised md:flex">
      <div className="border-b border-line px-5 py-5">
        <div className="text-2xs font-medium uppercase tracking-widest text-phase">
          KEIKO · TU Clausthal
        </div>
        <div className="mt-1 text-sm font-semibold leading-tight text-ink">
          Camera-only
          <br />
          heart rate
        </div>
        <p className="mt-2 text-2xs leading-relaxed text-ink-faint">
          Measuring a pulse from ordinary video, with no sensor touching the person.
        </p>
      </div>

      <nav className="flex-1 space-y-0.5 p-3">
        {NAV.map((n) => {
          const active = path === n.href || (n.href !== "/" && path.startsWith(n.href));
          return (
            <Link
              key={n.href}
              href={n.href}
              className={`block rounded-md px-3 py-2 transition-colors ${
                active
                  ? "bg-phase/10 text-phase"
                  : "text-ink-soft hover:bg-bg-panel hover:text-ink"
              }`}
            >
              <div className="text-sm font-medium">{n.label}</div>
              <div
                className={`text-2xs leading-tight ${
                  active ? "text-phase/70" : "text-ink-dim"
                }`}
              >
                {n.blurb}
              </div>
            </Link>
          );
        })}
      </nav>

      <div className="space-y-2 border-t border-line px-5 py-4 text-2xs text-ink-dim">
        <div className="flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-sm bg-phase" />
          <span className="text-ink-faint">PHASE-Net</span>
          <span className="ml-auto inline-block h-2 w-2 rounded-sm bg-pos" />
          <span className="text-ink-faint">POS</span>
        </div>
        <p className="leading-relaxed">
          Numbers are computed in Python; this page only draws them.
        </p>
        <a
          href="https://github.com/elhamBaradaran/rPPG"
          className="inline-block text-ink-faint underline underline-offset-2 hover:text-phase"
        >
          Source code
        </a>
      </div>
    </aside>
  );
}
