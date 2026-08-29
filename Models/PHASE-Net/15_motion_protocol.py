"""
A controlled motion protocol: at what level of movement does each method break?

WHY THIS REPLACES THE EARLIER TEST
  Two single-recording motion tests gave contradictory answers - in the first POS was
  rock-solid and PHASE-Net wobbled, in the second the reverse. The reason is that the
  motion intensity was never measured, only described ("gentle", "stronger"). A yes/no
  answer to "is it robust?" is therefore meaningless; the honest question is at what
  level of movement each method degrades.

  So instead of labelling the motion, this script MEASURES it, and plots error against
  the measured value. Two numbers describe each recording:

      displacement   how far the face centre moved, in pixels. Translation.
      appearance     mean absolute frame-to-frame pixel change inside the crop.
                     This captures what displacement misses - head rotation, changing
                     illumination across the face, motion blur, and facial movement
                     while talking.

  The second one matters because the previous experiment showed the face barely moved
  (21 % of a face width, correlation +0.07 with the error) while the model still failed.
  If appearance change tracks the error where displacement did not, that identifies the
  real mechanism.

EACH RECORDING IS A SANDWICH
  still -> condition -> still. Each condition therefore carries its own heart-rate
  baseline, so drift between recordings cannot contaminate the comparison.

USAGE
    python 15_motion_protocol.py --watch 83
    python 15_motion_protocol.py --labels still slow fast talk
"""

import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _common import CLIP_LEN, IMG_SIZE, HAAR, LARGE_BOX_COEF, load_model  # noqa: E402
from hr_methods import LOW_HZ, HIGH_HZ, clean                              # noqa: E402
from scipy.signal import periodogram                                        # noqa: E402

_DETECTOR = cv2.CascadeClassifier(HAAR)
ORDER = ["still", "slow", "talk", "lean", "fast", "full"]      # rough intensity order


# ---------------------------------------------------------------------------
# measuring how much actually moved
# ---------------------------------------------------------------------------
def appearance_change(crops, step=1):
    """Mean absolute frame-to-frame pixel change, per frame, in 0-255 units.

    Rotation, illumination shifts, blur and talking all change the image without
    necessarily moving the face - this sees all of them.
    """
    a = crops[:-step].astype(np.int16)
    b = crops[step:].astype(np.int16)
    d = np.abs(b - a).mean(axis=(1, 2, 3))
    return np.concatenate([[d[0]], d])            # pad to full length


def face_displacement(regions, box, every=15, smooth=0.6):
    """Distance of the detected face centre from where the static box was placed."""
    n = len(regions)
    c0 = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
    cur, out = None, np.zeros(n)
    for i in range(n):
        if i % every == 0:
            faces = _DETECTOR.detectMultiScale(regions[i])
            if len(faces) >= 1:
                b = faces[int(np.argmax(faces[:, 2]))].astype(float)
                b[0] -= (LARGE_BOX_COEF - 1.0) / 2 * b[2]
                b[1] -= (LARGE_BOX_COEF - 1.0) / 2 * b[3]
                b[2] *= LARGE_BOX_COEF
                b[3] *= LARGE_BOX_COEF
                c = np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
                cur = c if cur is None else smooth * c + (1 - smooth) * cur
        out[i] = np.linalg.norm(cur - c0) if cur is not None else 0.0
    return out


# ---------------------------------------------------------------------------
def crop_static(region, box):
    x, y, w, h = [int(round(v)) for v in box]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, region.shape[1]), min(y + h, region.shape[0])
    sub = region if (x1 - x0 < 8 or y1 - y0 < 8) else region[y0:y1, x0:x1]
    return cv2.resize(sub, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def phasenet_waveform(crops, model, device, step=64, tag=""):
    n = len(crops)
    starts = list(range(0, max(n - CLIP_LEN, 0) + 1, step))
    if starts and starts[-1] != n - CLIP_LEN:
        starts.append(n - CLIP_LEN)
    hann = np.hanning(CLIP_LEN) + 1e-6
    out, wsum = np.zeros(n), np.zeros(n)
    for i, a in enumerate(starts):
        x = np.transpose(crops[a:a + CLIP_LEN], (3, 0, 1, 2)).astype(np.float32)
        with torch.no_grad():
            pred, _ = model(torch.from_numpy(x).unsqueeze(0).to(device))
        w = pred[0].float().cpu().numpy()
        w = (w - w.mean()) / (w.std() + 1e-8)
        out[a:a + CLIP_LEN] += w * hann
        wsum[a:a + CLIP_LEN] += hann
        print(f"\r    {tag} clip {i+1}/{len(starts)}", end="", flush=True)
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
def analyse(path, model, device, win_s):
    z = np.load(path, allow_pickle=True)
    # The filename is authoritative. An early version of the recorder let the
    # display loop overwrite the label variable before saving, so some files carry
    # a phase caption ("SIT STILL AGAIN") instead of the condition name.
    label = re.sub(r"^motion_|\.npz$", "", os.path.basename(path))
    regions = z["regions"]
    fs = float(z["fps"][0])
    box = z["box_in_region"].astype(float)
    bounds = z["phase_bounds"].astype(float)
    n = len(regions)

    crops = np.asarray([crop_static(r, box) for r in regions])
    app = appearance_change(crops)
    disp = face_displacement(regions, box)

    waves = {
        "PHASE-Net": phasenet_waveform(crops, model, device, tag=f"{label:6s}"),
        "POS": pos_waveform(crops),
    }
    traces = {k: hr_trace(v, fs, win_s, 0.5) for k, v in waves.items()}

    tt = np.arange(n) / fs
    in_mid = (tt >= bounds[1]) & (tt < bounds[2])
    in_still = ((tt >= bounds[0]) & (tt < bounds[1])) | (tt >= bounds[2])

    rec = {
        "label": label, "file": os.path.basename(path), "fps": fs,
        "frames": n, "bounds": bounds.tolist(),
        "appearance_still": float(app[in_still].mean()),
        "appearance_motion": float(app[in_mid].mean()),
        "displacement_motion_max": float(disp[in_mid].max()),
        "face_width": float(box[2]),
        "methods": {},
    }
    # the motion "dose": how much more the image changed than at rest
    rec["motion_dose"] = rec["appearance_motion"] - rec["appearance_still"]

    for name, (t, hr) in traces.items():
        ok = np.isfinite(hr)
        still = ok & (((t >= bounds[0]) & (t < bounds[1])) | (t >= bounds[2]))
        mid = ok & (t >= bounds[1]) & (t < bounds[2])
        base = float(np.median(hr[still])) if still.any() else np.nan
        rec["methods"][name] = {
            "baseline_hr": base,
            "condition_hr": float(np.median(hr[mid])) if mid.any() else np.nan,
            "drift": float(np.mean(np.abs(hr[mid] - base))) if mid.any() else np.nan,
            "still_sd": float(np.std(hr[still])) if still.any() else np.nan,
            "condition_sd": float(np.std(hr[mid])) if mid.any() else np.nan,
            "condition_min": float(np.min(hr[mid])) if mid.any() else np.nan,
            "condition_max": float(np.max(hr[mid])) if mid.any() else np.nan,
        }
    return rec, traces, (tt, app, disp), bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="*", default=None,
                    help="which conditions to include; default is every motion_*.npz found")
    ap.add_argument("--watch", type=float, default=None)
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(HERE, "motion_*.npz")))
    if args.labels:
        want = set(args.labels)
        files = [f for f in files
                 if re.sub(r"^motion_|\.npz$", "", os.path.basename(f)) in want]

    # Older recordings stored pre-cropped 128x128 frames under a different key and have
    # no room for the face to move, so they cannot be part of this protocol.
    usable = []
    for f in files:
        keys = np.load(f, allow_pickle=True).files
        if "regions" in keys and "box_in_region" in keys:
            usable.append(f)
        else:
            print(f"  skipping {os.path.basename(f)} - older format, no saved region")
    files = usable

    if not files:
        sys.exit("No motion_*.npz recordings found. Record some with 13_record_full.py "
                 "--label still|slow|fast|talk")

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    model = load_model(device)
    print(f"{len(files)} recording(s) on {device}\n")

    records, extras = [], {}
    for f in files:
        print(f"  {os.path.basename(f)}")
        rec, traces, curves, bounds = analyse(f, model, device, args.win)
        records.append(rec)
        extras[rec["label"]] = (traces, curves, bounds)

    # sort by how much the image actually changed, not by the label we gave it
    records.sort(key=lambda r: r["motion_dose"])

    print("\n" + "=" * 78)
    print("  Conditions ordered by MEASURED motion, not by their name")
    print("=" * 78)
    print(f"{'condition':<10}{'motion dose':>12}{'disp max':>10}   "
          f"{'PHASE-Net drift':>16}{'POS drift':>11}")
    print("-" * 78)
    for r in records:
        p = r["methods"]["PHASE-Net"]["drift"]
        q = r["methods"]["POS"]["drift"]
        print(f"{r['label']:<10}{r['motion_dose']:>12.2f}"
              f"{r['displacement_motion_max']:>10.0f}   {p:>16.1f}{q:>11.1f}")
    print("\n  motion dose = extra frame-to-frame image change during the condition,")
    print("                in 0-255 pixel units. It sees rotation, lighting and talking,")
    print("                which displacement alone misses.")
    print("  drift       = mean |HR - that recording's own still baseline|, in BPM.")

    print(f"\n{'condition':<10}{'method':<12}{'baseline':>10}{'during':>9}"
          f"{'drift':>8}{'still sd':>10}{'cond sd':>9}{'range during':>16}")
    print("-" * 78)
    for r in records:
        for name, m in r["methods"].items():
            print(f"{r['label']:<10}{name:<12}{m['baseline_hr']:>10.1f}"
                  f"{m['condition_hr']:>9.1f}{m['drift']:>8.1f}{m['still_sd']:>10.2f}"
                  f"{m['condition_sd']:>9.2f}"
                  f"{m['condition_min']:>8.0f}-{m['condition_max']:<7.0f}")

    # ------------------------------------------------------------ conclusions
    print("\n" + "=" * 78)
    if args.watch:
        print(f"  smartwatch reference: {args.watch:.0f} BPM")
        for r in records:
            offs = ", ".join(f"{k} {m['baseline_hr']-args.watch:+.1f}"
                             for k, m in r["methods"].items())
            print(f"    {r['label']:<10} baseline offset: {offs}")

    doses = np.array([r["motion_dose"] for r in records], float)
    for name in ("PHASE-Net", "POS"):
        d = np.array([r["methods"][name]["drift"] for r in records], float)
        ok = np.isfinite(doses) & np.isfinite(d)
        if ok.sum() > 2 and np.std(doses[ok]) > 0:
            c = float(np.corrcoef(doses[ok], d[ok])[0, 1])
            print(f"\n  {name}: correlation between measured motion and drift = {c:+.2f}")
            if c > 0.7:
                print("    Strong - the error really is driven by how much the image changes.")
            elif c > 0.3:
                print("    Moderate - image change explains part of it, not all.")
            else:
                print("    Weak - something other than image change is driving the error.")

    if len(records) >= 2:
        base = next((r for r in records if r["label"] == "still"), None)
        if base:
            print(f"\n  Noise floor from the 'still' control: "
                  f"PHASE-Net {base['methods']['PHASE-Net']['drift']:.1f}, "
                  f"POS {base['methods']['POS']['drift']:.1f} BPM.")
            print("    Any condition below this is not distinguishable from doing nothing.")
        talk = next((r for r in records if r["label"] == "talk"), None)
        if talk:
            print(f"\n  'talk' moves the face very little "
                  f"({talk['displacement_motion_max']:.0f} px) but changes its appearance. "
                  f"\n    PHASE-Net drift {talk['methods']['PHASE-Net']['drift']:.1f}, "
                  f"POS {talk['methods']['POS']['drift']:.1f} BPM - if this is high, the "
                  f"\n    mechanism is appearance change, not the face leaving the crop.")
    print(f"\n  CAUTION: one subject, one recording per condition. Indicative only.")

    # ------------------------------------------------------------------- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [r["label"] for r in records]
        pn = [r["methods"]["PHASE-Net"]["drift"] for r in records]
        po = [r["methods"]["POS"]["drift"] for r in records]

        fig = plt.figure(figsize=(13, 4 + 2.6 * len(records)))
        gs = fig.add_gridspec(1 + len(records), 2, height_ratios=[2.2] + [1] * len(records))

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(doses, pn, "o-", color="#1b9e77", lw=2, ms=8, label="PHASE-Net")
        ax.plot(doses, po, "s-", color="#d95f02", lw=2, ms=8, label="POS")
        for x, y, l in zip(doses, pn, labels):
            ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
        ax.set_xlabel("measured motion (extra frame-to-frame change, 0-255 units)")
        ax.set_ylabel("drift from baseline (BPM)")
        ax.set_title("Error against how much the image actually changed")
        ax.legend(); ax.grid(alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        w = 0.36
        idx = np.arange(len(labels))
        ax2.bar(idx - w / 2, pn, w, color="#1b9e77", label="PHASE-Net")
        ax2.bar(idx + w / 2, po, w, color="#d95f02", label="POS")
        ax2.set_xticks(idx); ax2.set_xticklabels(labels)
        ax2.set_ylabel("drift (BPM)")
        ax2.set_title("Per condition")
        ax2.legend(); ax2.grid(alpha=0.3, axis="y")

        for i, r in enumerate(records):
            traces, (tt, app, disp), bounds = extras[r["label"]]
            a = fig.add_subplot(gs[1 + i, :])
            a.axvspan(bounds[1], bounds[2], color="orange", alpha=0.16)
            for name, colour in (("PHASE-Net", "#1b9e77"), ("POS", "#d95f02")):
                t, hr = traces[name]
                a.plot(t, hr, lw=1.5, color=colour, label=name)
            if args.watch:
                a.axhline(args.watch, color="k", ls="--", lw=1)
            a.set_ylabel("BPM")
            a.set_title(f"{r['label']}  -  motion dose {r['motion_dose']:.2f}", fontsize=10)
            a.grid(alpha=0.3)
            ab = a.twinx()
            ab.plot(tt, app, lw=0.8, color="#7570b3", alpha=0.55)
            ab.set_ylabel("image change", color="#7570b3", fontsize=8)
            if i == 0:
                a.legend(loc="upper right", fontsize=8)
            if i == len(records) - 1:
                a.set_xlabel("seconds")

        fig.tight_layout()
        out = os.path.join(HERE, "motion_protocol.png")
        fig.savefig(out, dpi=135)
        print(f"\n  plot -> {out}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    with open(os.path.join(HERE, "motion_protocol.json"), "w", encoding="utf-8") as f:
        json.dump({"watch_bpm": args.watch, "window_s": args.win, "records": records},
                  f, indent=2, default=float)
    print(f"  json -> {os.path.join(HERE, 'motion_protocol.json')}")


if __name__ == "__main__":
    main()
