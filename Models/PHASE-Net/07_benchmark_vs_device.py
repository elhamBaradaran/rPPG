"""
Benchmark HR-extraction methods against the OXIMETER'S OWN reported heart rate.

WHY THIS EXISTS
  The usual rPPG convention estimates the reference HR by running the same FFT
  pipeline over the ground-truth BVP waveform. That makes the "reference" a
  processed quantity - and when the processing gets it wrong, a CORRECT prediction
  is scored as an error.

  That is exactly what happened on subject47: our pipeline reported the model at
  114.3 BPM against a computed reference of 105.5 (an apparent 8.8 BPM failure),
  but the CMS50E itself reported 112.0 - so the model was right and the reference
  estimate was wrong.

  UBFC's ground_truth.txt line 2 holds the device's own HR readout. It requires no
  processing from us, so it can arbitrate between methods without favouring any.

WHAT IT REPORTS
  * convention error : |method(pred) - method(gt_bvp)|   - comparable to the paper
  * TRUE error       : |method(pred) - device HR|        - what actually matters

USAGE
    python 07_benchmark_vs_device.py --data "D:\\00-TU-CLAUSTHAL\\keiko-rppg\\UBFC"
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _common import device_hr_median                  # noqa: E402
from hr_methods import METHODS                        # noqa: E402

CACHE = os.path.join(HERE, "signal_cache.npz")


def load_cache(path=CACHE):
    if not os.path.isfile(path):
        sys.exit(f"No cache at {path}. Run 05_cache_signals.py first.")
    z = np.load(path, allow_pickle=True)
    ids = sorted({k.split("__")[0] for k in z.files},
                 key=lambda s: int(s.replace("subject", "")))
    return [{
        "id": sid,
        "pred": z[f"{sid}__pred"].astype(np.float64),
        "ref": z[f"{sid}__ref"].astype(np.float64),
        "fs": int(z[f"{sid}__meta"][1]),
        "split": str(z[f"{sid}__split"]),
    } for sid in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="UBFC root (for ground_truth.txt line 2)")
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    subs = load_cache(args.cache)
    fs = subs[0]["fs"]

    for s in subs:
        s["device_hr"] = device_hr_median(
            os.path.join(args.data, s["id"], "ground_truth.txt"))
    subs = [s for s in subs if np.isfinite(s["device_hr"])]
    held = [s for s in subs if s["split"] == "test"]

    print(f"{len(subs)} subject(s); {len(held)} truly held out")
    print("Reference = median of the CMS50E's own HR readout (independent of our code)\n")

    results = {}
    for name, fn in METHODS.items():
        per = {}
        for s in subs:
            try:
                pred = fn(s["pred"], fs)
                conv_ref = fn(s["ref"], fs)
            except Exception:
                pred = conv_ref = float("nan")
            per[s["id"]] = {
                "pred": pred,
                "conv_ref": conv_ref,
                "true_err": abs(pred - s["device_hr"]) if np.isfinite(pred) else np.nan,
                "conv_err": abs(pred - conv_ref) if np.isfinite(pred) and np.isfinite(conv_ref) else np.nan,
                "ref_err": abs(conv_ref - s["device_hr"]) if np.isfinite(conv_ref) else np.nan,
            }
        t_all = np.array([per[s["id"]]["true_err"] for s in subs], float)
        t_held = np.array([per[s["id"]]["true_err"] for s in held], float)
        c_held = np.array([per[s["id"]]["conv_err"] for s in held], float)
        r_all = np.array([per[s["id"]]["ref_err"] for s in subs], float)
        results[name] = {
            "per": per,
            "true_mae_held": float(np.nanmean(t_held)),
            "true_mae_all": float(np.nanmean(t_all)),
            "true_rmse_all": float(np.sqrt(np.nanmean(t_all ** 2))),
            "true_worst": float(np.nanmax(t_all)),
            "conv_mae_held": float(np.nanmean(c_held)),
            "ref_mae_all": float(np.nanmean(r_all)),
            "within3": float(np.nanmean(t_all <= 3.0) * 100),
            "within5": float(np.nanmean(t_all <= 5.0) * 100),
        }

    print("Scored against the DEVICE (the honest number):")
    print(f"{'method':<24}{'MAE held':>9}{'MAE all':>9}{'RMSE':>8}{'worst':>8}"
          f"{'<=3':>7}{'<=5':>7}   {'refErr':>7}")
    print("-" * 82)
    order = sorted(results, key=lambda k: results[k]["true_mae_held"])
    for name in order:
        r = results[name]
        tag = "  <- baseline" if name.startswith("baseline") else ""
        print(f"{name:<24}{r['true_mae_held']:>9.2f}{r['true_mae_all']:>9.2f}"
              f"{r['true_rmse_all']:>8.2f}{r['true_worst']:>8.1f}"
              f"{r['within3']:>7.0f}{r['within5']:>7.0f}   {r['ref_mae_all']:>7.2f}{tag}")
    print("\n  refErr = how wrong that method is at reading the GROUND-TRUTH signal's own HR.")
    print("  A large refErr means the method corrupts the reference, inflating or")
    print("  masking the apparent model error under the usual convention.")

    best = order[0]
    print(f"\nPer-subject: device HR vs baseline vs {best}")
    print(f"{'subject':<12}{'device':>8}{'baseline':>10}{'err':>7}   "
          f"{best[:12]:>12}{'err':>7}   split")
    print("-" * 76)
    for s in subs:
        b = results["baseline (repo)"]["per"][s["id"]]
        n = results[best]["per"][s["id"]]
        print(f"{s['id']:<12}{s['device_hr']:>8.1f}{b['pred']:>10.1f}{b['true_err']:>7.1f}   "
              f"{n['pred']:>12.1f}{n['true_err']:>7.1f}   {s['split']}")

    base = results["baseline (repo)"]
    print("\n" + "=" * 76)
    print("  Scored against the CMS50E device:")
    print(f"    baseline   MAE held {base['true_mae_held']:.2f}   all {base['true_mae_all']:.2f} BPM")
    print(f"    {best:<10} MAE held {results[best]['true_mae_held']:.2f}   "
          f"all {results[best]['true_mae_all']:.2f} BPM")
    print(f"\n  The convention (FFT on both signals) reports baseline held-out MAE "
          f"{base['conv_mae_held']:.2f},")
    print(f"  but its reference estimate is itself off by {base['ref_mae_all']:.2f} BPM on "
          f"average -")
    print("  so the conventional number is not a reliable measure of true accuracy.")
    if len(held) < 10:
        print(f"\n  CAUTION: {len(held)} held-out subjects only. Treat method ranking as "
              f"indicative.")

    out = os.path.join(HERE, "hr_benchmark_device.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per"} |
                   {"per_subject": v["per"]} for k, v in results.items()},
                  f, indent=2, default=float)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
