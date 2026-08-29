"""
The controlled test: is the motion failure caused by the fixed face box?

THE EXPERIMENT
  One recording, processed two ways. Everything is identical - the same frames, the same
  model, the same heart-rate extraction - except how the 128x128 crop is chosen:

      static    the face box found on the first frame, reused for all 90 seconds.
                This is what the training config specifies (DO_DYNAMIC_DETECTION: False)
                and what produced the earlier failure.

      dynamic   the face is re-detected every second inside the saved region, so the
                crop follows the head.

  Because the crop strategy is the only variable, a difference between the two is caused
  by it. If both fail equally, the hypothesis is wrong and the cause lies elsewhere -
  motion blur, changing illumination, or the model itself.

  A third measurement decides it directly: how far the face actually moved, over time.
  If the static run's errors line up with the displacement curve, the mechanism is
  confirmed rather than merely consistent.

USAGE
    python 14_static_vs_dynamic.py --watch 83
    python 14_static_vs_dynamic.py --detect-every 15     # denser re-detection
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _common import CLIP_LEN, IMG_SIZE, HAAR, LARGE_BOX_COEF, load_model  # noqa: E402
from hr_methods import LOW_HZ, HIGH_HZ, clean                              # noqa: E402
from scipy.signal import periodogram                                        # noqa: E402

REC = os.path.join(HERE, "motion_full.npz")

# One classifier, reused. _common.detect_face_box builds a new one per call, which reloads
# a 963 KB XML every time - fine for a handful of calls, wasteful for thousands.
_DETECTOR = cv2.CascadeClassifier(HAAR)


def detect_box(img_rgb):
    """Largest face, enlarged by the same 1.5x factor the training pipeline uses."""
    faces = _DETECTOR.detectMultiScale(img_rgb)
    if len(faces) < 1:
        return None
    b = list(faces[int(np.argmax(faces[:, 2]))]) if len(faces) >= 2 else list(faces[0])
    b[0] = max(0, b[0] - (LARGE_BOX_COEF - 1.0) / 2 * b[2])
    b[1] = max(0, b[1] - (LARGE_BOX_COEF - 1.0) / 2 * b[3])
    b[2] = LARGE_BOX_COEF * b[2]
    b[3] = LARGE_BOX_COEF * b[3]
    return np.asarray(b, dtype=float)


def crop_to_model(region, box):
    """Cut `box` out of a stored region and resize to the model's 128x128 input."""
    x, y, w, h = [int(round(v)) for v in box]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, region.shape[1]), min(y + h, region.shape[0])
    if x1 - x0 < 8 or y1 - y0 < 8:
        sub = region
    else:
        sub = region[y0:y1, x0:x1]
    return cv2.resize(sub, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
def track_face(regions, every, smooth=0.6):
    """Re-detect the face every `every` frames; hold and smooth in between.

    Returns the per-frame box and the per-frame detection status, so we can tell a
    genuine "the face moved" from a "the detector lost it".
    """
    n = len(regions)
    boxes = np.zeros((n, 4), dtype=float)
    found = np.zeros(n, dtype=bool)
    cur = None
    for i in range(n):
        if i % every == 0:
            b = detect_box(regions[i])
            if b is not None:
                cur = b if cur is None else smooth * b + (1 - smooth) * cur
                found[i] = True
        if cur is None:
            cur = np.array([0, 0, regions.shape[2], regions.shape[1]], dtype=float)
        boxes[i] = cur
    return boxes, found


def phasenet_waveform(get_clip, n, model, device, step=64, label=""):
    """Overlap-add the model's 128-frame clips into one continuous waveform."""
    starts = list(range(0, max(n - CLIP_LEN, 0) + 1, step))
    if starts and starts[-1] != n - CLIP_LEN:
        starts.append(n - CLIP_LEN)
    hann = np.hanning(CLIP_LEN) + 1e-6
    out, wsum = np.zeros(n), np.zeros(n)
    for i, a in enumerate(starts):
        clip = np.asarray([get_clip(j) for j in range(a, a + CLIP_LEN)])
        x = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32)      # 0-255, as trained
        with torch.no_grad():
            pred, _ = model(torch.from_numpy(x).unsqueeze(0).to(device))
        w = pred[0].float().cpu().numpy()
        w = (w - w.mean()) / (w.std() + 1e-8)
        out[a:a + CLIP_LEN] += w * hann
        wsum[a:a + CLIP_LEN] += hann
        print(f"\r  {label} clip {i+1}/{len(starts)}", end="", flush=True)
    print()
    return out / np.maximum(wsum, 1e-6)


def pos_waveform(crops):
    rgb = crops.reshape(len(crops), -1, 3).mean(axis=1)
    x = np.asarray(rgb, dtype=np.float64).T
    mean = x.mean(axis=1, keepdims=True)
    nrm = x / np.where(mean == 0, 1e-9, mean)
    s1 = nrm[1] - nrm[2]
    s2 = nrm[1] + nrm[2] - 2 * nrm[0]
    return s1 + (np.std(s1) / (np.std(s2) + 1e-9)) * s2


def window_hr(seg, fs, nfft=1 << 14):
    f, p = periodogram(seg - seg.mean(), fs=fs, nfft=nfft, detrend=False)
    m = (f >= LOW_HZ) & (f <= HIGH_HZ)
    return float(f[m][np.argmax(p[m])] * 60) if m.any() else float("nan")


def hr_trace(sig, fs, win_s=10.0, step_s=0.5):
    s = clean(np.asarray(sig, dtype=np.float64), fs)
    w, st = int(win_s * fs), max(int(step_s * fs), 1)
    t, hr = [], []
    for a in range(0, max(len(s) - w, 0) + 1, st):
        t.append((a + w / 2) / fs)
        hr.append(window_hr(s[a:a + w], fs))
    return np.array(t), np.array(hr)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default=REC)
    ap.add_argument("--watch", type=float, default=None)
    ap.add_argument("--detect-every", type=int, default=30,
                    help="frames between re-detections (30 = 1 s, the toolbox default)")
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.rec):
        sys.exit(f"No recording at {args.rec}. Run 13_record_full.py first.")
    z = np.load(args.rec, allow_pickle=True)
    regions = z["regions"]
    fs = float(z["fps"][0])
    static_box = z["box_in_region"].astype(float)
    names = [str(x) for x in z["phase_names"]]
    bounds = z["phase_bounds"].astype(float)
    n = len(regions)

    print(f"{n} frames at {fs:.1f} fps ({n/fs:.1f}s), region {regions.shape[2]}x{regions.shape[1]}")
    print("phases: " + ", ".join(f"{p} [{bounds[i]:.0f}-{bounds[i+1]:.0f}s]"
                                 for i, p in enumerate(names)))
    print(f"static box in region: x={static_box[0]:.0f} y={static_box[1]:.0f} "
          f"w={static_box[2]:.0f} h={static_box[3]:.0f}\n")

    # --------------------------------------------------------------- tracking
    print(f"Tracking the face (re-detecting every {args.detect_every} frames)...")
    boxes, found = track_face(regions, args.detect_every)
    det_frames = int(found.sum())
    attempts = int(np.ceil(n / args.detect_every))
    print(f"  face located in {det_frames}/{attempts} detection attempts "
          f"({100*det_frames/max(attempts,1):.0f}%)")

    # how far the face centre wandered from where it started
    c0 = np.array([static_box[0] + static_box[2] / 2, static_box[1] + static_box[3] / 2])
    centres = np.stack([boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
    disp = np.linalg.norm(centres - c0, axis=1)
    face_w = static_box[2]
    print(f"  displacement: median {np.median(disp):.0f} px, max {disp.max():.0f} px "
          f"({disp.max()/face_w*100:.0f}% of a face width)")

    # did the face ever leave the crop the static run is stuck with?
    left = np.mean(disp > face_w * 0.25) * 100
    print(f"  face centre more than a quarter face-width off: {left:.0f}% of frames")

    # --------------------------------------------------------------- inference
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    model = load_model(device)

    static_crops = np.asarray([crop_to_model(r, static_box) for r in regions])
    print("\nRunning the model twice on the same frames:")
    waves = {
        "PHASE-Net static": phasenet_waveform(lambda j: static_crops[j], n, model,
                                              device, label="static "),
        "PHASE-Net dynamic": phasenet_waveform(lambda j: crop_to_model(regions[j], boxes[j]),
                                               n, model, device, label="dynamic"),
        "POS static": pos_waveform(static_crops),
    }
    traces = {k: hr_trace(v, fs, args.win, 0.5) for k, v in waves.items()}

    # ----------------------------------------------------------------- metrics
    def mask(t, i):
        return (t >= bounds[i]) & (t < bounds[i + 1])

    print(f"\n{'method':<20}{'still HR':>10}{'motion HR':>11}{'drift':>8}"
          f"{'still sd':>10}{'motion sd':>11}")
    print("-" * 70)
    results = {}
    for name, (t, hr) in traces.items():
        ok = np.isfinite(hr)
        still = ok & (mask(t, 0) | mask(t, 2))
        motion = ok & mask(t, 1)
        base = float(np.median(hr[still])) if still.any() else np.nan
        results[name] = {
            "still_hr": base,
            "motion_hr": float(np.median(hr[motion])) if motion.any() else np.nan,
            "drift": float(np.mean(np.abs(hr[motion] - base))) if motion.any() else np.nan,
            "still_sd": float(np.std(hr[still])) if still.any() else np.nan,
            "motion_sd": float(np.std(hr[motion])) if motion.any() else np.nan,
            "motion_min": float(np.min(hr[motion])) if motion.any() else np.nan,
            "motion_max": float(np.max(hr[motion])) if motion.any() else np.nan,
        }
        r = results[name]
        print(f"{name:<20}{base:>10.1f}{r['motion_hr']:>11.1f}{r['drift']:>8.1f}"
              f"{r['still_sd']:>10.2f}{r['motion_sd']:>11.2f}")

    # does the static run's error follow the displacement curve?
    t_s, hr_s = traces["PHASE-Net static"]
    disp_t = np.interp(t_s, np.arange(n) / fs, disp)
    err_s = np.abs(hr_s - results["PHASE-Net static"]["still_hr"])
    ok = np.isfinite(err_s)
    corr = (float(np.corrcoef(disp_t[ok], err_s[ok])[0, 1])
            if ok.sum() > 3 and np.std(disp_t[ok]) > 0 else float("nan"))

    # ----------------------------------------------------------------- verdict
    st, dy = results["PHASE-Net static"], results["PHASE-Net dynamic"]
    print("\n" + "=" * 70)
    print(f"  correlation between face displacement and static-crop error: {corr:+.2f}")
    print(f"    (a strong positive value means the errors happen exactly when the face moves)")
    print(f"\n  drift during motion:  static {st['drift']:.1f}  ->  dynamic {dy['drift']:.1f} BPM")
    if np.isfinite(st["drift"]) and np.isfinite(dy["drift"]):
        if dy["drift"] < st["drift"] * 0.6:
            print("  HYPOTHESIS SUPPORTED: tracking the face largely fixes the failure,")
            print("  so the static crop - not the architecture - was the problem.")
        elif dy["drift"] < st["drift"] * 0.9:
            print("  PARTIALLY SUPPORTED: tracking helps but does not fully fix it;")
            print("  motion blur or changing illumination likely contribute as well.")
        else:
            print("  HYPOTHESIS NOT SUPPORTED: tracking the face changed little, so the")
            print("  cause lies elsewhere - blur, illumination, or the model itself.")
    if left > 5 and disp.max() > face_w * 0.5:
        print("\n  NOTE: the face left the saved region at times, so even the dynamic run")
        print("        may be cropping incomplete faces. A larger --region-scale would help.")
    print("  CAUTION: one subject, one recording.")

    # -------------------------------------------------------------------- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colours = {"PHASE-Net static": "#d62728", "PHASE-Net dynamic": "#1b9e77",
                   "POS static": "#d95f02"}
        fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1, 1]})
        for i, p in enumerate(names):
            if "MOVE" in p.upper():
                for a in ax:
                    a.axvspan(bounds[i], bounds[i + 1], color="orange", alpha=0.16)
        for name, (t, hr) in traces.items():
            ax[0].plot(t, hr, lw=1.8, label=name, color=colours[name])
        if args.watch:
            ax[0].axhline(args.watch, color="k", ls="--", lw=1.2,
                          label=f"smartwatch {args.watch:.0f}")
        ax[0].set_ylabel("heart rate (BPM)")
        ax[0].set_title("Same recording, same model - only the crop strategy differs")
        ax[0].legend(loc="upper right")
        ax[0].grid(alpha=0.3)

        tt = np.arange(n) / fs
        ax[1].plot(tt, disp, lw=1.4, color="#7570b3")
        ax[1].axhline(face_w * 0.25, color="grey", ls=":", lw=1,
                      label="quarter of a face width")
        ax[1].set_ylabel("face displacement (px)")
        ax[1].set_title("How far the face moved from where the static box was placed")
        ax[1].legend(loc="upper right")
        ax[1].grid(alpha=0.3)

        ax[2].plot(t_s, err_s, lw=1.5, color=colours["PHASE-Net static"],
                   label="static-crop error")
        ax[2].plot(t_s, disp_t / max(disp_t.max(), 1e-9) * np.nanmax(err_s),
                   lw=1.2, ls="--", color="#7570b3", label="displacement (scaled)")
        ax[2].set_xlabel("seconds")
        ax[2].set_ylabel("|HR - baseline|")
        ax[2].set_title(f"Do they move together?   correlation {corr:+.2f}")
        ax[2].legend(loc="upper right")
        ax[2].grid(alpha=0.3)

        fig.tight_layout()
        out = os.path.join(HERE, "static_vs_dynamic.png")
        fig.savefig(out, dpi=140)
        print(f"\n  plot -> {out}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    with open(os.path.join(HERE, "static_vs_dynamic.json"), "w", encoding="utf-8") as f:
        json.dump({"fps": fs, "phases": names, "bounds": bounds.tolist(),
                   "watch_bpm": args.watch, "detect_every": args.detect_every,
                   "displacement_px": {"median": float(np.median(disp)),
                                       "max": float(disp.max()),
                                       "face_width": float(face_w)},
                   "displacement_error_correlation": corr,
                   "results": results}, f, indent=2, default=float)
    print(f"  json -> {os.path.join(HERE, 'static_vs_dynamic.json')}")


if __name__ == "__main__":
    main()
