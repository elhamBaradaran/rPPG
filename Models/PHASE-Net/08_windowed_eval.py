"""
Windowed evaluation - the protocol this task actually calls for.

WHY THE PREVIOUS PROTOCOL WAS WRONG
  One heart rate per 60-second video assumes the heart rate is constant. It is not.
  On subject47 the oximeter itself reports 103-120 BPM within a single recording,
  so the whole-video spectrum is genuinely bimodal: 105.5 and 114.3 are BOTH real.
  Taking argmax then compares whichever mode happened to dominate in the prediction
  against whichever dominated in the reference - and calls the difference "error".

  The fix is not better peak picking. It is to stop summarising a time-varying
  quantity with a single number.

WHAT THIS DOES
  Slides a window over the recording, estimates HR in each window (where the signal
  is close to stationary), and scores three ways:

    1. paired windows   - model HR vs reference HR in the SAME window   (the honest
                          measure of tracking, and the one with real statistics:
                          hundreds of points instead of 15)
    2. video median     - median of windowed HRs, model vs reference    (comparable
                          to how rPPG papers report per-video MAE)
    3. vs the device    - model HR vs the CMS50E's own readout in that window
                          (fully independent of our signal processing)

  The filter is applied ONCE to the whole signal and windows are cut afterwards, so
  filtfilt edge transients do not contaminate every window.

USAGE
    python 08_windowed_eval.py --data "D:\\00-TU-CLAUSTHAL\\keiko-rppg\\UBFC"
    python 08_windowed_eval.py --data ... --win 10 --step 1 --estimator welch
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from hr_methods import LOW_HZ, HIGH_HZ, clean            # noqa: E402
from scipy.signal import periodogram, welch              # noqa: E402

CACHE = os.path.join(HERE, "signal_cache.npz")
CLIP_LEN = 128


# ---------------------------------------------------------------------------
# per-window HR estimators (input is ALREADY filtered)
# ---------------------------------------------------------------------------
def est_periodogram(seg, fs, nfft=1 << 14):
    f, p = periodogram(seg - seg.mean(), fs=fs, nfft=nfft, detrend=False)
    m = (f >= LOW_HZ) & (f <= HIGH_HZ)
    return float(f[m][np.argmax(p[m])] * 60)


def est_welch(seg, fs, nfft=1 << 14):
    nper = max(int(len(seg) / 2), 32)
    f, p = welch(seg - seg.mean(), fs=fs, nperseg=nper, noverlap=nper // 2,
                 nfft=nfft, detrend=False)
    m = (f >= LOW_HZ) & (f <= HIGH_HZ)
    if not m.any():
        return float("nan")
    return float(f[m][np.argmax(p[m])] * 60)


ESTIMATORS = {"periodogram": est_periodogram, "welch": est_welch}


def zscore_clips(x, clip=CLIP_LEN):
    """Normalise each model output clip before it is concatenated.

    The model runs on independent 128-frame clips; joining them raw leaves steps at
    every boundary. That boundary rate (30/128 = 0.234 Hz) has harmonics inside the
    search band - 112.5 BPM among them - so it can manufacture a spectral peak.
    """
    y = np.asarray(x, dtype=np.float64).copy()
    for a in range(0, len(y) - clip + 1, clip):
        seg = y[a:a + clip]
        s = seg.std()
        y[a:a + clip] = (seg - seg.mean()) / (s if s > 1e-12 else 1.0)
    return y


def windows(n, fs, win_s, step_s):
    w, st = int(win_s * fs), int(step_s * fs)
    return [(a, a + w) for a in range(0, max(n - w, 0) + 1, st)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--estimator", choices=list(ESTIMATORS), default="periodogram")
    ap.add_argument("--zscore-clips", action="store_true",
                    help="normalise each 128-frame clip before analysis")
    args = ap.parse_args()

    est = ESTIMATORS[args.estimator]
    z = np.load(args.cache, allow_pickle=True)
    ids = sorted({k.split("__")[0] for k in z.files},
                 key=lambda s: int(s.replace("subject", "")))

    print(f"window {args.win:g}s, step {args.step:g}s, estimator '{args.estimator}'"
          f"{', per-clip z-score' if args.zscore_clips else ''}\n")
    print(f"{'subject':<11}{'wins':>5}{'pairMAE':>9}{'medMdl':>8}{'medRef':>8}"
          f"{'medErr':>8}{'devHR':>7}{'devErr':>8}   split")
    print("-" * 78)

    rows, all_pair, all_dev = [], [], []
    for sid in ids:
        pred = z[f"{sid}__pred"].astype(np.float64)
        ref = z[f"{sid}__ref"].astype(np.float64)
        fs = int(z[f"{sid}__meta"][1])
        split = str(z[f"{sid}__split"])

        if args.zscore_clips:
            pred = zscore_clips(pred, CLIP_LEN)

        n = min(len(pred), len(ref))
        pred, ref = pred[:n], ref[:n]

        # filter the FULL signal once, then window (avoids per-window edge transients)
        fp, fr = clean(pred, fs), clean(ref, fs)

        # the device's own HR, sample-aligned with the BVP
        gt_path = os.path.join(args.data, sid, "ground_truth.txt")
        dev = None
        if os.path.isfile(gt_path):
            with open(gt_path) as f:
                lines = [l for l in f.read().split("\n") if l.strip()]
            if len(lines) >= 2:
                d = np.array([float(x) for x in lines[1].split()])
                dev = d[:n] if len(d) >= n else np.pad(d, (0, n - len(d)), mode="edge")

        pair_err, mdl_hrs, ref_hrs, dev_err = [], [], [], []
        for a, b in windows(n, fs, args.win, args.step):
            hm, hr_ = est(fp[a:b], fs), est(fr[a:b], fs)
            if np.isfinite(hm) and np.isfinite(hr_):
                pair_err.append(abs(hm - hr_))
                mdl_hrs.append(hm)
                ref_hrs.append(hr_)
                if dev is not None:
                    dw = dev[a:b]
                    dw = dw[(dw >= 40) & (dw <= 180)]
                    if dw.size:
                        dev_err.append(abs(hm - float(np.median(dw))))
        if not pair_err:
            continue

        med_m, med_r = float(np.median(mdl_hrs)), float(np.median(ref_hrs))
        dev_med = float(np.median(dev[(dev >= 40) & (dev <= 180)])) if dev is not None else np.nan
        row = {
            "id": sid, "split": split, "n_windows": len(pair_err),
            "pair_mae": float(np.mean(pair_err)),
            "median_model": med_m, "median_ref": med_r,
            "median_err": abs(med_m - med_r),
            "device_hr": dev_med,
            "device_mae_windowed": float(np.mean(dev_err)) if dev_err else float("nan"),
        }
        rows.append(row)
        all_pair.extend(pair_err)
        all_dev.extend(dev_err)
        print(f"{sid:<11}{row['n_windows']:>5}{row['pair_mae']:>9.2f}"
              f"{med_m:>8.1f}{med_r:>8.1f}{row['median_err']:>8.2f}"
              f"{dev_med:>7.0f}{row['device_mae_windowed']:>8.2f}   {split}")

    held = [r for r in rows if r["split"] == "test"]

    def agg(rs, key):
        v = np.array([r[key] for r in rs], dtype=float)
        return float(np.nanmean(v)), float(np.sqrt(np.nanmean(v ** 2)))

    print("\n" + "=" * 78)
    print(f"  {len(rows)} subjects, {len(all_pair)} windows total "
          f"({len(held)} subjects held out)")
    print("=" * 78)
    for label, rs in [("ALL", rows), ("HELD-OUT", held)]:
        if not rs:
            continue
        p_mae, p_rmse = agg(rs, "pair_mae")
        m_mae, m_rmse = agg(rs, "median_err")
        d_mae, _ = agg(rs, "device_mae_windowed")
        print(f"\n  {label} ({len(rs)} subjects)")
        print(f"    paired-window MAE (model vs reference) : {p_mae:6.2f} BPM  "
              f"RMSE {p_rmse:5.2f}")
        print(f"    video median MAE  (model vs reference) : {m_mae:6.2f} BPM  "
              f"RMSE {m_rmse:5.2f}")
        print(f"    windowed MAE vs the CMS50E device      : {d_mae:6.2f} BPM")

    print(f"\n  paper reports 0.15 BPM on UBFC (12-subject test split)")
    print("  NOTE: 'vs reference' applies identical processing to both signals, which")
    print("        is the convention papers use. 'vs device' is independent of our")
    print("        code and is the stricter, more honest figure.")

    out = os.path.join(HERE, "windowed_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "subjects": rows}, f, indent=2, default=float)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
