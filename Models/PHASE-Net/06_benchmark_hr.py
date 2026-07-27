"""
Benchmark heart-rate extraction methods on the cached PHASE-Net signals.

Because 05_cache_signals.py already stored the model's raw output per subject,
every method here is scored on EXACTLY the same signals - so differences are due
to the extraction strategy alone, not to the model or preprocessing.

The reference HR is computed from the ground-truth BVP with the toolbox's own
baseline method, so "error" means the same thing as in the original evaluation.

IMPORTANT
  Only 6 subjects are truly held out of the checkpoint's training split, and only
  one of them fails today. Picking a method purely because it fixes that one
  subject would be overfitting, so this script reports held-out AND all-subject
  results, plus the per-subject breakdown, and flags methods that fix one case
  while breaking others.

USAGE
    python 06_benchmark_hr.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hr_methods                                    # noqa: E402
from hr_methods import METHODS, hr_baseline          # noqa: E402

CACHE = os.path.join(HERE, "signal_cache.npz")


def load_cache(path=CACHE):
    if not os.path.isfile(path):
        sys.exit(f"No cache at {path}. Run 05_cache_signals.py first.")
    z = np.load(path, allow_pickle=True)
    ids = sorted({k.split("__")[0] for k in z.files},
                 key=lambda s: int(s.replace("subject", "")))
    out = []
    for sid in ids:
        meta = z[f"{sid}__meta"]
        out.append({
            "id": sid,
            "pred": z[f"{sid}__pred"].astype(np.float64),
            "ref": z[f"{sid}__ref"].astype(np.float64),
            "clips": int(meta[0]),
            "fs": int(meta[1]),
            "split": str(z[f"{sid}__split"]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    subs = load_cache(args.cache)
    fs = subs[0]["fs"]
    held = [s for s in subs if s["split"] == "test"]
    print(f"{len(subs)} subject(s) cached; {len(held)} truly held out "
          f"({[s['id'] for s in held]})\n")

    # FAIRNESS: apply each method to BOTH the prediction and the reference BVP,
    # exactly as the toolbox does. Scoring every method against a reference HR
    # computed by the BASELINE would rig the comparison - predictions and
    # reference would share the baseline's coarse FFT grid (30/2048*60 = 0.88 BPM
    # per bin) and land in the same bin, showing a flattering "0.00" error that
    # finer methods cannot produce.
    for s in subs:
        s["hr_ref_baseline"] = hr_baseline(s["ref"], fs)

    results = {}
    for name, fn in METHODS.items():
        t0 = time.time()
        errs, per = [], {}
        for s in subs:
            try:
                hr = fn(s["pred"], fs)
                ref = fn(s["ref"], fs)                 # same method on both signals
            except Exception as e:
                print(f"  ! {name} failed on {s['id']}: {type(e).__name__}")
                hr = ref = float("nan")
            e = abs(hr - ref) if (np.isfinite(hr) and np.isfinite(ref)) else float("nan")
            per[s["id"]] = (hr, e, ref)
            errs.append(e)
        results[name] = {"per": per, "secs": time.time() - t0}

        e_all = np.array([v[1] for v in per.values()], dtype=float)
        e_held = np.array([per[s["id"]][1] for s in held], dtype=float)
        # "agree" = within one baseline FFT bin, the finest the status quo can resolve
        results[name]["agree_held"] = int(np.sum(e_held <= 0.88))
        results[name]["mae_all"] = float(np.nanmean(e_all))
        results[name]["rmse_all"] = float(np.sqrt(np.nanmean(e_all ** 2)))
        results[name]["mae_held"] = float(np.nanmean(e_held))
        results[name]["worst"] = float(np.nanmax(e_all))
        results[name]["exact_held"] = int(np.sum(e_held < 0.5))
        results[name]["within3_all"] = float(np.nanmean(e_all <= 3.0) * 100)

    # ------------------------------------------------------------------ table
    print(f"{'method':<24}{'MAE held':>9}{'MAE all':>9}{'RMSE all':>10}"
          f"{'worst':>8}{'agree/6':>9}{'<=3bpm':>8}{'sec':>7}")
    print("-" * 84)
    order = sorted(results, key=lambda k: (results[k]["mae_held"], results[k]["mae_all"]))
    for name in order:
        r = results[name]
        star = "  <- baseline" if name.startswith("baseline") else ""
        print(f"{name:<24}{r['mae_held']:>9.2f}{r['mae_all']:>9.2f}{r['rmse_all']:>10.2f}"
              f"{r['worst']:>8.1f}{r['agree_held']:>9d}{r['within3_all']:>8.1f}"
              f"{r['secs']:>7.1f}{star}")

    # --------------------------------------------------- per-subject detail
    base = results["baseline (repo)"]
    best = order[0]
    print(f"\nPer-subject: baseline vs best ({best})   [ref = same method on the GT signal]")
    print(f"{'subject':<12}{'baseRef':>9}{'basePred':>9}{'err':>7}   "
          f"{'bestRef':>9}{'bestPred':>9}{'err':>7}   split")
    print("-" * 84)
    for s in subs:
        b_hr, b_e, b_ref = base["per"][s["id"]]
        n_hr, n_e, n_ref = results[best]["per"][s["id"]]
        mark = ""
        if n_e < b_e - 0.5:
            mark = "  IMPROVED"
        elif n_e > b_e + 0.5:
            mark = "  WORSE"
        print(f"{s['id']:<12}{b_ref:>9.1f}{b_hr:>9.1f}{b_e:>7.1f}   "
              f"{n_ref:>9.1f}{n_hr:>9.1f}{n_e:>7.1f}   {s['split']:<5}{mark}")

    # ------------------------------------------------------------- honesty
    print("\n" + "=" * 68)
    improved = sum(1 for s in subs
                   if results[best]["per"][s["id"]][1] < base["per"][s["id"]][1] - 0.5)
    worse = sum(1 for s in subs
                if results[best]["per"][s["id"]][1] > base["per"][s["id"]][1] + 0.5)
    print(f"  NOTE: the baseline quantises HR to {30/2048*60:.2f} BPM bins, so its "
          f"'0.00' errors\n        mean 'same bin', not exact agreement.")
    print(f"  best method       : {best}")
    print(f"  MAE held-out      : {base['mae_held']:.2f} -> {results[best]['mae_held']:.2f} BPM")
    print(f"  MAE all subjects  : {base['mae_all']:.2f} -> {results[best]['mae_all']:.2f} BPM")
    print(f"  subjects improved : {improved}   worsened: {worse}")
    if worse:
        print("  CAUTION: it trades one failure for another - not a clean win.")
    if len(held) < 10:
        print(f"  CAUTION: only {len(held)} held-out subjects. Choosing a method on so few\n"
              f"           samples risks overfitting; prefer a method that is also\n"
              f"           principled, not merely the lowest number here.")

    out = os.path.join(HERE, "hr_benchmark.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per"} |
                   {"per_subject": {s: {"hr": t[0], "err": t[1], "ref": t[2]} for s, t in v["per"].items()}}
                   for k, v in results.items()}, f, indent=2)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
