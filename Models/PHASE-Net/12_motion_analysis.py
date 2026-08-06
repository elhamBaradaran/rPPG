"""
Analyse the motion test: does PHASE-Net hold up under head movement, and does POS?

THE MEASUREMENT, WITHOUT AN OXIMETER
  Heart rate barely changes across 90 seconds, so the two still phases give us a
  reference for the middle one. For each method we take its own median heart rate
  during the still phases as its baseline, then ask how far it wanders while the head
  is moving. A method that reports 75 BPM at rest and 40 BPM during motion has failed,
  and we can see that without any external sensor.

FAIRNESS
  Both methods produce a continuous pulse waveform from the SAME cropped frames, and
  both waveforms then go through the SAME heart-rate extraction (10-second windows,
  as established on UBFC). Only the algorithm that produces the waveform differs.

  PHASE-Net is constrained to 128-frame clips, so its clips are overlap-added with a
  Hann taper into one continuous signal - the same approach as 03_webcam_phasenet_v2.

USAGE
    python 12_motion_analysis.py
    python 12_motion_analysis.py --watch 74     # draw your smartwatch reading too
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _common import CLIP_LEN, load_model                     # noqa: E402
from hr_methods import LOW_HZ, HIGH_HZ, clean                # noqa: E402
from scipy.signal import periodogram                          # noqa: E402

REC = os.path.join(HERE, "motion_test.npz")


def window_hr(seg, fs, nfft=1 << 14):
    f, p = periodogram(seg - seg.mean(), fs=fs, nfft=nfft, detrend=False)
    m = (f >= LOW_HZ) & (f <= HIGH_HZ)
    return float(f[m][np.argmax(p[m])] * 60) if m.any() else float("nan")


def hr_trace(sig, fs, win_s=10.0, step_s=0.5):
    """Continuous waveform -> (time, HR) series, the protocol validated on UBFC."""
    s = clean(np.asarray(sig, dtype=np.float64), fs)
    w, st = int(win_s * fs), max(int(step_s * fs), 1)
    t, hr = [], []
    for a in range(0, max(len(s) - w, 0) + 1, st):
        t.append((a + w / 2) / fs)          # timestamp at the window centre
        hr.append(window_hr(s[a:a + w], fs))
    return np.array(t), np.array(hr)


def phasenet_waveform(crops, model, device, step=64):
    """Run the model over overlapping 128-frame clips and blend them into one signal."""
    n = len(crops)
    starts = list(range(0, max(n - CLIP_LEN, 0) + 1, step))
    if starts and starts[-1] != n - CLIP_LEN:
        starts.append(n - CLIP_LEN)
    hann = np.hanning(CLIP_LEN) + 1e-6
    out = np.zeros(n)
    wsum = np.zeros(n)
    for i, a in enumerate(starts):
        clip = crops[a:a + CLIP_LEN]
        x = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32)     # 0-255, as trained
        with torch.no_grad():
            pred, _ = model(torch.from_numpy(x).unsqueeze(0).to(device))
        w = pred[0].float().cpu().numpy()
        w = (w - w.mean()) / (w.std() + 1e-8)
        out[a:a + CLIP_LEN] += w * hann
        wsum[a:a + CLIP_LEN] += hann
        if (i + 1) % 10 == 0 or i == len(starts) - 1:
            print(f"\r  PHASE-Net clip {i+1}/{len(starts)}", end="", flush=True)
    print()
    return out / np.maximum(wsum, 1e-6)


def pos_waveform(rgb):
    """POS projection over the whole recording (Wang et al. maths)."""
    x = np.asarray(rgb, dtype=np.float64).T
    mean = x.mean(axis=1, keepdims=True)
    nrm = x / np.where(mean == 0, 1e-9, mean)
    s1 = nrm[1] - nrm[2]
    s2 = nrm[1] + nrm[2] - 2 * nrm[0]
    return s1 + (np.std(s1) / (np.std(s2) + 1e-9)) * s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default=REC)
    ap.add_argument("--watch", type=float, default=None,
                    help="smartwatch BPM, drawn as an independent reference")
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.rec):
        sys.exit(f"No recording at {args.rec}. Run 11_live_motion_test.py first.")
    z = np.load(args.rec, allow_pickle=True)
    crops = z["crops"]
    fs = float(z["fps"][0])
    names = [str(x) for x in z["phase_names"]]
    bounds = z["phase_bounds"].astype(float)

    print(f"{len(crops)} frames at {fs:.1f} fps ({len(crops)/fs:.1f}s)")
    print(f"phases: " + ", ".join(f"{n} [{bounds[i]:.0f}-{bounds[i+1]:.0f}s]"
                                  for i, n in enumerate(names)) + "\n")

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    model = load_model(device)

    rgb = crops.reshape(len(crops), -1, 3).mean(axis=1)
    waves = {
        "PHASE-Net": phasenet_waveform(crops, model, device),
        "POS": pos_waveform(rgb),
    }

    traces = {k: hr_trace(v, fs, args.win, 0.5) for k, v in waves.items()}

    # ------------------------------------------------------------- metrics
    def mask(t, i):
        return (t >= bounds[i]) & (t < bounds[i + 1])

    print(f"\n{'method':<12}{'still HR':>10}{'motion HR':>11}{'drift':>8}"
          f"{'still sd':>10}{'motion sd':>11}")
    print("-" * 62)
    results = {}
    for name, (t, hr) in traces.items():
        ok = np.isfinite(hr)
        still = ok & (mask(t, 0) | mask(t, 2))
        motion = ok & mask(t, 1)
        base = float(np.median(hr[still])) if still.any() else np.nan
        m_med = float(np.median(hr[motion])) if motion.any() else np.nan
        drift = float(np.mean(np.abs(hr[motion] - base))) if motion.any() else np.nan
        results[name] = {
            "still_hr": base, "motion_hr": m_med, "motion_drift": drift,
            "still_sd": float(np.std(hr[still])) if still.any() else np.nan,
            "motion_sd": float(np.std(hr[motion])) if motion.any() else np.nan,
            "motion_min": float(np.min(hr[motion])) if motion.any() else np.nan,
            "motion_max": float(np.max(hr[motion])) if motion.any() else np.nan,
        }
        r = results[name]
        print(f"{name:<12}{base:>10.1f}{m_med:>11.1f}{drift:>8.1f}"
              f"{r['still_sd']:>10.2f}{r['motion_sd']:>11.2f}")

    print("\n  still HR   = each method's own baseline, from the two still phases")
    print("  drift      = mean |HR - baseline| while the head was moving  <- the test")
    print("  sd         = how much the estimate jitters within a phase")
    if args.watch:
        print(f"\n  smartwatch reference: {args.watch:.0f} BPM")
        for name in traces:
            print(f"    {name:<12} still baseline off by "
                  f"{abs(results[name]['still_hr']-args.watch):5.1f} BPM")

    # ------------------------------------------------------------- verdict
    p, q = results["PHASE-Net"], results["POS"]
    print("\n" + "=" * 62)
    if np.isfinite(p["motion_drift"]) and np.isfinite(q["motion_drift"]):
        if q["motion_drift"] > p["motion_drift"] * 1.5:
            print(f"  PHASE-Net held up better under motion: drift "
                  f"{p['motion_drift']:.1f} vs {q['motion_drift']:.1f} BPM")
        elif p["motion_drift"] > q["motion_drift"] * 1.5:
            print(f"  POS held up better under motion: drift "
                  f"{q['motion_drift']:.1f} vs {p['motion_drift']:.1f} BPM")
        else:
            print(f"  Both behaved similarly under motion "
                  f"({p['motion_drift']:.1f} vs {q['motion_drift']:.1f} BPM drift) - "
                  f"the motion may not have been strong enough to separate them.")
    print(f"  POS range during motion       : "
          f"{q['motion_min']:.0f} - {q['motion_max']:.0f} BPM")
    print(f"  PHASE-Net range during motion : "
          f"{p['motion_min']:.0f} - {p['motion_max']:.0f} BPM")
    print("  CAUTION: one subject, one recording. Indicative, not conclusive.")

    # ------------------------------------------------------------- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
        for i, n in enumerate(names):
            if "MOVE" in n.upper():
                for a in ax:
                    a.axvspan(bounds[i], bounds[i + 1], color="orange", alpha=0.16)
        colours = {"PHASE-Net": "#1b9e77", "POS": "#d95f02"}
        for name, (t, hr) in traces.items():
            ax[0].plot(t, hr, lw=1.8, label=name, color=colours[name])
            ax[0].axhline(results[name]["still_hr"], color=colours[name],
                          ls=":", lw=1, alpha=0.7)
        if args.watch:
            ax[0].axhline(args.watch, color="k", ls="--", lw=1.2,
                          label=f"smartwatch {args.watch:.0f}")
        ax[0].set_ylabel("heart rate (BPM)")
        ax[0].set_title("Heart rate during still / motion / still   "
                        "(shaded = head moving)")
        ax[0].legend(loc="upper right")
        ax[0].grid(alpha=0.3)

        t0 = np.arange(len(waves["PHASE-Net"])) / fs
        for name, w in waves.items():
            ax[1].plot(t0, clean(w, fs) / (np.std(clean(w, fs)) + 1e-9),
                       lw=0.7, color=colours[name], alpha=0.85, label=name)
        ax[1].set_xlabel("seconds")
        ax[1].set_ylabel("pulse (normalised)")
        ax[1].set_title("Recovered pulse waveforms")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(HERE, "motion_test.png")
        fig.savefig(out, dpi=140)
        print(f"\n  plot -> {out}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    with open(os.path.join(HERE, "motion_results.json"), "w", encoding="utf-8") as f:
        json.dump({"fps": fs, "phases": names, "bounds": bounds.tolist(),
                   "watch_bpm": args.watch, "results": results},
                  f, indent=2, default=float)
    print(f"  json -> {os.path.join(HERE, 'motion_results.json')}")


if __name__ == "__main__":
    main()
