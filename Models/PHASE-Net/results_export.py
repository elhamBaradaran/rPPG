"""
Export rPPG evaluation results as JSON for the web dashboard.

DESIGN
  Python is the source of truth; the web portal only reads. Every run writes a
  self-contained JSON file to  <repo>/results/  so the dashboard can render both
  PHASE-Net and POS results from one place.

WHAT GOES IN, AND WHY
  * run metadata  - traceability: which weights (sha256), which git commit, which
                    library versions, which preprocessing. A regulated medical
                    product must be able to answer "how exactly was this number
                    produced?" months later.
  * metrics       - the four standard rPPG numbers (MAE / RMSE / MAPE / Pearson r)
                    PLUS clinical agreement statistics (Bland-Altman bias and
                    limits of agreement) and the +/-3 and +/-5 BPM pass rates.
  * per subject   - signed error (needed for Bland-Altman), quality signals.
  * waveforms     - predicted vs reference signal + spectrum, for the viewer.

This module is model-agnostic: POS can import it later and write to the same folder.
"""

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

SCHEMA_VERSION = "1.0"

# <repo>/results  (this file lives at <repo>/Models/PHASE-Net/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _safe(x):
    """Make a value JSON-legal. JSON has no NaN/Infinity - they become null."""
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if (np.isnan(x) or np.isinf(x)) else x
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_safe(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {k: _safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe(v) for v in x]
    return x


def file_sha256(path, chunk=1 << 20):
    """Hash a file so a result can always be traced back to exact weights."""
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit(repo_dir=REPO_ROOT):
    """Current commit of the analysis code, or None outside a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def environment_info(device=None):
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    except Exception:
        pass
    for mod in ("numpy", "scipy", "cv2"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def compute_metrics(subjects):
    """Aggregate metrics from a list of per-subject dicts.

    Each subject needs at least 'hr_ref' and 'hr_pred'.

    Bland-Altman is the standard way clinicians judge whether a new measurement
    method agrees with a reference: it looks at the DIFFERENCE (bias) and the
    range within which 95% of differences fall (limits of agreement), rather than
    correlation - two methods can correlate perfectly and still disagree badly.
    """
    # Accept either the raw evaluator naming ('hr_ref') or the exported dashboard
    # naming ('hr_ref_bpm'), so callers can pass whichever they already have.
    def _get(s, *names):
        for n in names:
            if n in s and s[n] is not None:
                return s[n]
        return float("nan")

    ref = np.array([_get(s, "hr_ref_bpm", "hr_ref") for s in subjects], dtype=float)
    pred = np.array([_get(s, "hr_pred_bpm", "hr_pred") for s in subjects], dtype=float)
    ok = ~(np.isnan(ref) | np.isnan(pred))
    ref, pred = ref[ok], pred[ok]
    n = len(ref)
    if n == 0:
        return {}

    diff = pred - ref                     # signed, for bias
    err = np.abs(diff)

    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs(diff / ref)) * 100) if np.all(ref != 0) else float("nan")

    # Pearson r needs at least 2 points and non-zero variance
    if n >= 2 and np.std(ref) > 0 and np.std(pred) > 0:
        pearson = float(np.corrcoef(ref, pred)[0, 1])
    else:
        pearson = float("nan")

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n >= 2 else float("nan")

    snrs = [s["snr_db"] for s in subjects
            if s.get("snr_db") is not None and not np.isnan(s["snr_db"])]
    maccs = [s["macc"] for s in subjects
             if s.get("macc") is not None and not np.isnan(s["macc"])]

    return {
        "n_subjects": int(n),
        "mae_bpm": mae,
        "rmse_bpm": rmse,
        "mape_pct": mape,
        "pearson_r": pearson,
        # clinical agreement
        "bland_altman": {
            "bias_bpm": bias,
            "sd_bpm": sd,
            "loa_lower_bpm": float(bias - 1.96 * sd) if not np.isnan(sd) else None,
            "loa_upper_bpm": float(bias + 1.96 * sd) if not np.isnan(sd) else None,
        },
        # AAMI-style acceptability bands
        "within_3bpm_pct": float(np.mean(err <= 3.0) * 100),
        "within_5bpm_pct": float(np.mean(err <= 5.0) * 100),
        "mean_snr_db": float(np.mean(snrs)) if snrs else None,
        "mean_macc": float(np.mean(maccs)) if maccs else None,
    }


# ---------------------------------------------------------------------------
# waveforms
# ---------------------------------------------------------------------------
def waveform_payload(subject_id, pred, ref, fs, max_seconds=60, decimals=5):
    """Prepare one subject's signals + spectrum for the waveform viewer.

    Both signals are z-scored so they overlay on the same axis (rPPG output has
    arbitrary units; only the SHAPE and timing carry meaning).
    """
    from scipy.signal import periodogram

    n = int(min(len(pred), len(ref), max_seconds * fs))
    pred = np.asarray(pred[:n], dtype=float)
    ref = np.asarray(ref[:n], dtype=float)

    def z(x):
        s = np.std(x)
        return (x - np.mean(x)) / (s if s > 1e-12 else 1.0)

    pz, rz = z(pred), z(ref)

    def psd(x):
        f, p = periodogram(x - np.mean(x), fs=fs, nfft=4096, detrend=False)
        band = (f >= 0.7) & (f <= 3.0)          # 42-180 BPM
        pb = p[band]
        pb = pb / (pb.max() if pb.max() > 0 else 1.0)
        return (f[band] * 60.0), pb             # x axis in BPM

    fbpm, ppsd = psd(pz)
    _, rpsd = psd(rz)

    return {
        "subject": subject_id,
        "fs": fs,
        "n_samples": n,
        "duration_s": round(n / fs, 2),
        "signals": {
            "predicted": [round(float(v), decimals) for v in pz],
            "reference": [round(float(v), decimals) for v in rz],
        },
        "spectrum": {
            "bpm": [round(float(v), 2) for v in fbpm],
            "predicted": [round(float(v), 4) for v in ppsd],
            "reference": [round(float(v), 4) for v in rpsd],
        },
    }


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------
def write_results(run_id, model, dataset, preprocessing, hr_extraction,
                  subjects, waveforms=None, device=None, notes=None,
                  results_dir=RESULTS_DIR):
    """Write one evaluation run as JSON. Returns the path written."""
    os.makedirs(results_dir, exist_ok=True)
    wf_dir = os.path.join(results_dir, "waveforms")

    metrics = compute_metrics(subjects)

    wf_index = {}
    if waveforms:
        os.makedirs(wf_dir, exist_ok=True)
        for sid, payload in waveforms.items():
            fname = f"{run_id}__{sid}.json"
            with open(os.path.join(wf_dir, fname), "w", encoding="utf-8") as f:
                json.dump(_safe(payload), f)
            wf_index[sid] = f"waveforms/{fname}"

    doc = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "id": run_id,
            "created_utc": datetime.datetime.now(datetime.timezone.utc)
                            .replace(microsecond=0).isoformat(),
            "git_commit": git_commit(),
            "environment": environment_info(device),
            "notes": notes,
        },
        "model": model,
        "dataset": dataset,
        "preprocessing": preprocessing,
        "hr_extraction": hr_extraction,
        "metrics": metrics,
        "subjects": subjects,
        "waveforms": wf_index,
    }

    path = os.path.join(results_dir, f"{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_safe(doc), f, indent=2)

    _update_index(results_dir, run_id, doc)
    return path


def _update_index(results_dir, run_id, doc):
    """Maintain results/index.json so the dashboard can list all runs."""
    idx_path = os.path.join(results_dir, "index.json")
    index = {"schema_version": SCHEMA_VERSION, "runs": []}
    if os.path.isfile(idx_path):
        try:
            with open(idx_path, encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            pass

    entry = {
        "id": run_id,
        "file": f"{run_id}.json",
        "created_utc": doc["run"]["created_utc"],
        "model": doc["model"].get("name"),
        "dataset": doc["dataset"].get("name"),
        "n_subjects": doc["metrics"].get("n_subjects"),
        "mae_bpm": doc["metrics"].get("mae_bpm"),
        "rmse_bpm": doc["metrics"].get("rmse_bpm"),
        "pearson_r": doc["metrics"].get("pearson_r"),
    }
    index["runs"] = [r for r in index.get("runs", []) if r.get("id") != run_id]
    index["runs"].append(entry)
    index["runs"].sort(key=lambda r: r.get("created_utc") or "", reverse=True)

    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(_safe(index), f, indent=2)
