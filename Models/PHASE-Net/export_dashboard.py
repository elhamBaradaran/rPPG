"""
Assemble every result into one payload the dashboard can read.

WHY
  The numbers currently live in seven JSON files scattered across two folders, each
  shaped for the script that wrote it. A web app should not have to know any of that.
  This script gathers them into a single, stable document with one job: being easy to
  render.

  The principle stays as before - Python is the source of truth, the web layer only
  displays. Every figure in the dashboard traces back to a script in this folder, and
  the payload carries the weights hash, git commit and library versions that produced it.

OUTPUT
  results/dashboard.json      everything except the waveforms (small, ~40 KB)
  results/waveforms/*.json    already written by results_export.py, referenced by name

USAGE
    python export_dashboard.py
"""

import datetime
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(REPO, "results")
sys.path.insert(0, HERE)

SCHEMA = "1.0"


def load(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def latest_ubfc_run():
    """The most recent UBFC evaluation written by results_export.py."""
    idx = load(os.path.join(RESULTS, "index.json"), {})
    runs = idx.get("runs") or []
    if not runs:
        return None
    return load(os.path.join(RESULTS, runs[0]["file"]))


def build_validation(run):
    """UBFC validation: metrics, per-subject rows, and the traceability record.

    The held-out flag matters more than any single number here: the released checkpoint
    was trained on the first 72 % of subjects, so only the rest measure generalisation.
    """
    if not run:
        return None
    try:
        from _common import train_test_split
        _, test_split = train_test_split()
    except Exception:
        test_split = []

    subjects = []
    for s in run.get("subjects", []):
        subjects.append({
            "id": s["id"],
            "held_out": s["id"] in test_split,
            "hr_reference": s.get("hr_ref_bpm"),
            "hr_predicted": s.get("hr_pred_bpm"),
            "error": s.get("error_bpm"),                 # signed, for Bland-Altman
            "abs_error": s.get("abs_error_bpm"),
            "mean_hr": s.get("mean_hr_bpm"),             # Bland-Altman x axis
            "snr_db": s.get("snr_db"),
            "macc": s.get("macc"),
            "duration_s": s.get("duration_s"),
            "waveform": run.get("waveforms", {}).get(s["id"]),
        })

    windowed = load(os.path.join(HERE, "windowed_eval.json"), {})
    w_subjects = {r["id"]: r for r in windowed.get("subjects", [])}
    for s in subjects:
        w = w_subjects.get(s["id"])
        if w:
            s["windowed"] = {
                "paired_mae": w.get("pair_mae"),
                "median_model": w.get("median_model"),
                "median_reference": w.get("median_ref"),
                "device_hr": w.get("device_hr"),
                "device_mae": w.get("device_mae_windowed"),
                "n_windows": w.get("n_windows"),
            }

    held = [s for s in subjects if s["held_out"]]

    def mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None

    return {
        "dataset": run.get("dataset"),
        "model": run.get("model"),
        "preprocessing": run.get("preprocessing"),
        "hr_extraction": run.get("hr_extraction"),
        "metrics_video_level": run.get("metrics"),
        "metrics_windowed": {
            "protocol": windowed.get("config"),
            "held_out": {
                "paired_mae": mean([s.get("windowed", {}).get("paired_mae") for s in held]),
                "device_mae": mean([s.get("windowed", {}).get("device_mae") for s in held]),
                "n_subjects": len(held),
            },
            "all": {
                "paired_mae": mean([s.get("windowed", {}).get("paired_mae") for s in subjects]),
                "device_mae": mean([s.get("windowed", {}).get("device_mae") for s in subjects]),
                "n_subjects": len(subjects),
            },
            "total_windows": sum(s.get("windowed", {}).get("n_windows", 0) for s in subjects),
        },
        "subjects": subjects,
        "n_held_out": len(held),
    }


def build_comparison():
    """PHASE-Net against both POS implementations, same crops and same protocol."""
    d = load(os.path.join(HERE, "model_comparison.json"))
    if not d:
        return None
    methods = d.get("methods", [])
    rows = []
    for s in d.get("subjects", []):
        row = {"id": s["id"], "held_out": s.get("split") == "test",
               "device_hr": s.get("device_hr"), "methods": {}}
        for m in methods:
            v = s.get("methods", {}).get(m, {})
            row["methods"][m] = {
                "median_hr": v.get("median_hr"),
                "mae_vs_reference": v.get("mae_vs_ref"),
                "mae_vs_device": v.get("mae_vs_device"),
            }
        rows.append(row)

    def agg(rs, m, key):
        vals = [r["methods"][m][key] for r in rs
                if isinstance(r["methods"].get(m, {}).get(key), (int, float))]
        return {"mean": sum(vals) / len(vals) if vals else None,
                "worst": max(vals) if vals else None}

    held = [r for r in rows if r["held_out"]]
    summary = {m: {"held_out": {k: agg(held, m, k)
                                for k in ("mae_vs_reference", "mae_vs_device")},
                   "all": {k: agg(rows, m, k)
                           for k in ("mae_vs_reference", "mae_vs_device")}}
               for m in methods}
    return {"methods": methods, "protocol": d.get("config"),
            "summary": summary, "subjects": rows,
            "n_held_out": len(held), "n_all": len(rows)}


def build_motion():
    """The controlled protocol, plus the hypotheses it ruled out."""
    d = load(os.path.join(HERE, "motion_protocol.json"))
    if not d:
        return None
    conditions = []
    for r in d.get("records", []):
        conditions.append({
            "label": r["label"],
            "motion_dose": r.get("motion_dose"),
            "displacement_max_px": r.get("displacement_motion_max"),
            "face_width_px": r.get("face_width"),
            "duration_s": round(r.get("frames", 0) / max(r.get("fps", 30), 1), 1),
            "methods": {k: {"baseline_hr": v.get("baseline_hr"),
                            "condition_hr": v.get("condition_hr"),
                            "drift": v.get("drift"),
                            "still_sd": v.get("still_sd"),
                            "condition_sd": v.get("condition_sd"),
                            "range": [v.get("condition_min"), v.get("condition_max")]}
                        for k, v in r.get("methods", {}).items()},
        })
    conditions.sort(key=lambda c: c["motion_dose"] if c["motion_dose"] is not None else 0)

    sd = load(os.path.join(HERE, "static_vs_dynamic.json"), {})
    control = next((c for c in conditions if c["label"] == "still"), None)
    talk = next((c for c in conditions if c["label"] == "talk"), None)

    return {
        "watch_bpm": d.get("watch_bpm"),
        "window_s": d.get("window_s"),
        "conditions": conditions,
        "noise_floor": {k: v["drift"] for k, v in (control or {}).get("methods", {}).items()},
        "hypotheses_tested": [
            {
                "hypothesis": "The face leaves the fixed crop under motion",
                "test": "One recording processed twice - fixed box versus re-detecting "
                        "the face every second - so the crop strategy is the only variable",
                "result": "rejected",
                "evidence": {
                    "max_displacement_fraction_of_face_width":
                        round((sd.get("displacement_px", {}).get("max", 0) /
                               max(sd.get("displacement_px", {}).get("face_width", 1), 1)), 3),
                    "displacement_error_correlation":
                        sd.get("displacement_error_correlation"),
                    "drift_static": sd.get("results", {}).get("PHASE-Net static", {}).get("drift"),
                    "drift_dynamic": sd.get("results", {}).get("PHASE-Net dynamic", {}).get("drift"),
                },
                "note": "Tracking made it worse: a re-detected box jitters between frames "
                        "and injects motion of its own. A slightly misaligned but stable "
                        "crop beats a well-centred jittery one.",
            },
            {
                "hypothesis": "Facial appearance change (speech, expression) breaks it",
                "test": "A 'talk' condition - the face changes continuously but barely moves",
                "result": "rejected",
                "evidence": {
                    "talk_drift": (talk or {}).get("methods", {}).get("PHASE-Net", {}).get("drift"),
                    "still_drift": (control or {}).get("methods", {}).get("PHASE-Net", {}).get("drift"),
                    "talk_displacement_px": (talk or {}).get("displacement_max_px"),
                },
                "note": "Talking is indistinguishable from doing nothing.",
            },
            {
                "hypothesis": "One method is inherently more motion-robust",
                "test": "Dose-response across five measured motion levels",
                "result": "rejected",
                "note": "Both degrade by a similar factor and the effect saturates - "
                        "tripling the motion from slow to fast changes nothing.",
            },
        ],
    }


def build_extraction_benchmark():
    """Nine HR-extraction strategies, scored against both references."""
    conv = load(os.path.join(HERE, "hr_benchmark.json"), {})
    dev = load(os.path.join(HERE, "hr_benchmark_device.json"), {})
    if not conv and not dev:
        return None
    methods = sorted(set(conv) | set(dev))
    rows = []
    for m in methods:
        rows.append({
            "method": m,
            "vs_reference": {"mae_held_out": conv.get(m, {}).get("mae_held"),
                             "mae_all": conv.get(m, {}).get("mae_all"),
                             "worst": conv.get(m, {}).get("worst")},
            "vs_device": {"mae_held_out": dev.get(m, {}).get("true_mae_held"),
                          "mae_all": dev.get(m, {}).get("true_mae_all"),
                          "reference_error": dev.get(m, {}).get("ref_mae_all")},
        })
    rows.sort(key=lambda r: r["vs_device"]["mae_held_out"]
              if isinstance(r["vs_device"]["mae_held_out"], (int, float)) else 1e9)
    return {
        "methods": rows,
        "note": "Scoring a method against itself applied to the reference measures "
                "self-consistency, not accuracy - one method looked 18x better that way "
                "than against the oximeter's own readout. 'reference_error' is how wrong "
                "each method is at reading the ground-truth signal's own heart rate.",
    }


def main():
    run = latest_ubfc_run()
    validation = build_validation(run)
    comparison = build_comparison()
    motion = build_motion()
    benchmark = build_extraction_benchmark()

    held = (validation or {}).get("metrics_windowed", {}).get("held_out", {})
    noise = (motion or {}).get("noise_floor", {})

    payload = {
        "schema_version": SCHEMA,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                          .replace(microsecond=0).isoformat(),
        "project": {
            "title": "Camera-only heart rate monitoring for human-robot collaboration",
            "context": "KEIKO - Kognitiv und Empathisch Intelligente Kollaborierende Roboter",
            "context_url": "https://www.simzentrum.de/forschungsprojekte/keiko/",
            "institution": "TU Clausthal, with the University of Gottingen",
            "repository": "https://github.com/elhamBaradaran/rPPG",
        },
        "headline": {
            "held_out_mae_vs_reference": held.get("paired_mae"),
            "held_out_mae_vs_device": held.get("device_mae"),
            "n_held_out_subjects": held.get("n_subjects"),
            "total_windows": (validation or {}).get("metrics_windowed", {}).get("total_windows"),
            "webcam_noise_floor": noise.get("PHASE-Net"),
            "paper_claim": 0.15,
        },
        # Stated up front rather than buried. A dashboard that only shows its best
        # numbers is a sales page, not a result.
        "limitations": [
            "Only 6 subjects are genuinely held out of the released checkpoint's "
            "training split; the other 9 measure memorisation, not generalisation.",
            "Half of the paper's 12-subject test split was unavailable, so the "
            "comparison with its reported 0.15 BPM is not like-for-like.",
            "On this webcam the noise floor is about 8 BPM while sitting perfectly "
            "still - twenty times the error achieved on the controlled UBFC recordings. "
            "Capture quality, not the model, is the current bottleneck.",
            "The motion protocol is one subject, one recording per condition.",
            "Ground truth is not always right: on two subjects the oximeter's own HR "
            "readout is wrong by more than 20 BPM, confirmed by two independent "
            "analyses of its waveform.",
        ],
        "validation": validation,
        "comparison": comparison,
        "motion": motion,
        "extraction_benchmark": benchmark,
        "traceability": {
            "run": (run or {}).get("run"),
            "model": (run or {}).get("model"),
            "note": "Every figure here is produced by a script in Models/PHASE-Net. "
                    "The weights hash, git commit and library versions below identify "
                    "exactly what produced them.",
        },
    }

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "dashboard.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)

    n_wave = len(glob.glob(os.path.join(RESULTS, "waveforms", "*.json")))
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} KB)")
    print(f"sections: " + ", ".join(k for k in
          ("validation", "comparison", "motion", "extraction_benchmark")
          if payload.get(k)))
    missing = [k for k in ("validation", "comparison", "motion", "extraction_benchmark")
               if not payload.get(k)]
    if missing:
        print(f"missing (source JSON not found): {', '.join(missing)}")
    print(f"waveform files available: {n_wave}")
    h = payload["headline"]
    print(f"\nheadline: held-out MAE {h['held_out_mae_vs_reference']:.2f} BPM vs reference, "
          f"{h['held_out_mae_vs_device']:.2f} vs device, over {h['total_windows']} windows")
    print(f"          webcam noise floor {h['webcam_noise_floor']:.1f} BPM")


if __name__ == "__main__":
    main()
