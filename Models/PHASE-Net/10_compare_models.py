"""
PHASE-Net vs the POS baseline, under one identical protocol.

WHAT MAKES THIS A FAIR COMPARISON
  Every method here saw the same videos, the same Haar face box, the same 1.5x
  enlargement and the same 128x128 crop. They are scored with the same 10-second
  windowed protocol, against the same two references. The only thing that differs
  is the algorithm that turns those frames into a pulse.

THREE METHODS
  PHASE-Net    the deep model (from signal_cache.npz)
  POS ref      the repository's own POS_WANG - Wang et al. (2017) with a 1.6 s
               sliding window; the implementation the paper compares against
  POS ours     the simplified POS written from scratch in Models/POS

TWO REFERENCES, because neither alone is trustworthy
  vs reference  the same processing applied to the ground-truth BVP waveform. This
                is the convention rPPG papers use, so it is comparable to published
                numbers - but shared processing can hide shared errors.
  vs device     the CMS50E's own HR readout, which involves none of our code. The
                stricter figure. Note two subjects (25, 27) have a faulty device
                readout, confirmed earlier by beat-counting; they are in the train
                split, so the held-out numbers are unaffected.

USAGE
    python 10_compare_models.py --data "D:\\00-TU-CLAUSTHAL\\keiko-rppg\\UBFC"
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from hr_methods import LOW_HZ, HIGH_HZ, clean          # noqa: E402
from scipy.signal import periodogram                    # noqa: E402

PHASE_CACHE = os.path.join(HERE, "signal_cache.npz")
POS_CACHE = os.path.join(HERE, "pos_cache.npz")


# ---------------------------------------------------------------------------
# the windowed protocol - identical to 08_windowed_eval.py
# ---------------------------------------------------------------------------
def window_hr(seg, fs, nfft=1 << 14):
    f, p = periodogram(seg - seg.mean(), fs=fs, nfft=nfft, detrend=False)
    m = (f >= LOW_HZ) & (f <= HIGH_HZ)
    return float(f[m][np.argmax(p[m])] * 60) if m.any() else float("nan")


def hr_series(sig, fs, win_s=10.0, step_s=1.0):
    """Filter the whole signal once, then slide a window over it.

    Filtering before windowing (rather than per window) keeps filtfilt's edge
    transients out of every single window.
    """
    s = clean(np.asarray(sig, dtype=np.float64), fs)
    w, st = int(win_s * fs), int(step_s * fs)
    out = []
    for a in range(0, max(len(s) - w, 0) + 1, st):
        out.append((a, a + w, window_hr(s[a:a + w], fs)))
    return out


def device_series(path, n, fs):
    """The oximeter's own HR, sample-aligned with the video."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        lines = [l for l in f.read().split("\n") if l.strip()]
    if len(lines) < 2:
        return None
    d = np.array([float(x) for x in lines[1].split()])
    return d[:n] if len(d) >= n else np.pad(d, (0, n - len(d)), mode="edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=1.0)
    args = ap.parse_args()

    for p in (PHASE_CACHE, POS_CACHE):
        if not os.path.isfile(p):
            sys.exit(f"Missing {os.path.basename(p)} - run 05_cache_signals.py "
                     f"and 09_pos_baseline.py first.")
    zp = np.load(PHASE_CACHE, allow_pickle=True)
    zq = np.load(POS_CACHE, allow_pickle=True)

    ids = sorted({k.split("__")[0] for k in zp.files},
                 key=lambda s: int(s.replace("subject", "")))
    ids = [s for s in ids if f"{s}__pos_ref" in zq.files]

    METHODS = ["PHASE-Net", "POS ref", "POS ours"]
    rows = []

    print(f"window {args.win:g}s, step {args.step:g}s   |   {len(ids)} subjects\n")
    print(f"{'subject':<11}{'device':>7} | " +
          " | ".join(f"{m:>9}" for m in METHODS) + "   split")
    print("-" * 62)

    for sid in ids:
        fs = int(zp[f"{sid}__meta"][1])
        split = str(zp[f"{sid}__split"])
        ref_bvp = zp[f"{sid}__ref"].astype(np.float64)
        sigs = {
            "PHASE-Net": zp[f"{sid}__pred"].astype(np.float64),
            "POS ref": zq[f"{sid}__pos_ref"].astype(np.float64),
            "POS ours": zq[f"{sid}__pos_ours"].astype(np.float64),
        }
        n = min([len(ref_bvp)] + [len(v) for v in sigs.values()])
        dev = device_series(os.path.join(args.data, sid, "ground_truth.txt"), n, fs)

        ref_hr = hr_series(ref_bvp[:n], fs, args.win, args.step)
        dev_med = np.nan
        if dev is not None:
            v = dev[(dev >= 40) & (dev <= 180)]
            dev_med = float(np.median(v)) if v.size else np.nan

        row = {"id": sid, "split": split, "device_hr": dev_med, "methods": {}}
        for name, sig in sigs.items():
            ser = hr_series(sig[:n], fs, args.win, args.step)
            k = min(len(ser), len(ref_hr))
            pair, dvs, hrs = [], [], []
            for i in range(k):
                a, b, hm = ser[i]
                _, _, hr_ = ref_hr[i]
                if np.isfinite(hm):
                    hrs.append(hm)
                    if np.isfinite(hr_):
                        pair.append(abs(hm - hr_))
                    if dev is not None:
                        dw = dev[a:b]
                        dw = dw[(dw >= 40) & (dw <= 180)]
                        if dw.size:
                            dvs.append(abs(hm - float(np.median(dw))))
            row["methods"][name] = {
                "median_hr": float(np.median(hrs)) if hrs else np.nan,
                "mae_vs_ref": float(np.mean(pair)) if pair else np.nan,
                "mae_vs_device": float(np.mean(dvs)) if dvs else np.nan,
                "n_windows": len(hrs),
            }
        rows.append(row)
        cells = " | ".join(f"{row['methods'][m]['mae_vs_ref']:>9.2f}" for m in METHODS)
        print(f"{sid:<11}{dev_med:>7.0f} | {cells}   {split}")

    held = [r for r in rows if r["split"] == "test"]

    def agg(rs, m, key):
        v = np.array([r["methods"][m][key] for r in rs], dtype=float)
        return float(np.nanmean(v)), float(np.nanmax(v))

    print("\n" + "=" * 68)
    print(f"  RESULTS   ({len(rows)} subjects, {len(held)} truly held out)")
    print("=" * 68)
    for label, rs in [("HELD-OUT (the number that counts)", held), ("ALL SUBJECTS", rows)]:
        if not rs:
            continue
        print(f"\n  {label} - n={len(rs)}")
        print(f"    {'method':<12}{'MAE vs ref':>12}{'worst':>8}   "
              f"{'MAE vs device':>14}{'worst':>8}")
        print("    " + "-" * 56)
        for m in METHODS:
            r_mae, r_worst = agg(rs, m, "mae_vs_ref")
            d_mae, d_worst = agg(rs, m, "mae_vs_device")
            print(f"    {m:<12}{r_mae:>12.2f}{r_worst:>8.1f}   "
                  f"{d_mae:>14.2f}{d_worst:>8.1f}")

    # ------------------------------------------------------------- verdict
    h_ref = {m: agg(held, m, "mae_vs_ref")[0] for m in METHODS}
    h_dev = {m: agg(held, m, "mae_vs_device")[0] for m in METHODS}
    print("\n" + "-" * 68)
    if h_ref["POS ref"] > 0:
        print(f"  PHASE-Net vs POS (reference implementation), held-out:")
        print(f"    vs reference : {h_ref['PHASE-Net']:.2f} vs {h_ref['POS ref']:.2f} BPM"
              f"   -> {h_ref['POS ref']/max(h_ref['PHASE-Net'],1e-9):.1f}x better")
        print(f"    vs device    : {h_dev['PHASE-Net']:.2f} vs {h_dev['POS ref']:.2f} BPM"
              f"   -> {h_dev['POS ref']/max(h_dev['PHASE-Net'],1e-9):.1f}x better")
    print(f"\n  Our own POS vs the reference POS, held-out (vs device):")
    print(f"    ours {h_dev['POS ours']:.2f} BPM  |  reference {h_dev['POS ref']:.2f} BPM")
    print("\n  CAVEAT: UBFC subjects are largely still, so this does NOT capture POS's")
    print("          known collapse under head motion. That comparison needs motion")
    print("          data (MMPD) or our own recordings.")
    print(f"  CAVEAT: {len(held)} held-out subjects only - treat the ratio as indicative.")

    out = os.path.join(HERE, "model_comparison.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "methods": METHODS, "subjects": rows},
                  f, indent=2, default=float)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
