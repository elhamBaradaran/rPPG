"""
Milestone 3 - Validate PHASE-Net on the UBFC-rPPG dataset (REAL ground truth).

This is the scientifically meaningful test: UBFC videos are clean 30 fps AND come
with a reference pulse from a CMS50E oximeter, so we can compute a real error
(MAE, in BPM) and compare it with the paper's reported 0.15 BPM.

It stays lightweight - it imports ONLY the model class and the repo's post-
processing (no TensorFlow / retinaface), and reproduces the toolbox's evaluation
faithfully:
  * face crop (Haar, static first frame, 1.5x box) -> 128x128, pixels 0-255
  * cut the video into non-overlapping 128-frame clips
  * run the model on each clip, concatenate the outputs into one per-video signal
  * feed that + the reference pulse to the repo's own calculate_metric_per_video()
    (detrend -> band-pass -> FFT), which returns predicted HR, reference HR, SNR

USAGE
  # point it at the UBFC root that contains subject1/, subject2/, ...
  python 04_ubfc_eval.py --data "D:\\path\\to\\UBFC"
  python 04_ubfc_eval.py --data "D:\\path\\to\\UBFC" --limit 3     # first 3 subjects
  python 04_ubfc_eval.py --selftest                                # no data needed
"""

import argparse
import csv
import datetime
import glob
import os
import re
import sys
import time

import cv2
import numpy as np
import torch

REPO = r"D:\00-TU-CLAUSTHAL\keiko-rppg\PhaseNet"
sys.path.insert(0, REPO)

WEIGHTS = os.path.join(REPO, "weights", "phasenet_ubfc_epoch9.pth")
HAAR = os.path.join(REPO, "dataset", "haarcascade_frontalface_default.xml")
HERE = os.path.dirname(os.path.abspath(__file__))

from neural_methods.model.PhaseNet.PhaseNet import PhaseNet             # noqa: E402
from evaluation.post_process import calculate_metric_per_video         # noqa: E402

sys.path.insert(0, HERE)
import results_export                                                  # noqa: E402

CLIP_LEN = 128
IMG_SIZE = 128
FS = 30                       # UBFC-rPPG is 30 fps
LARGE_BOX_COEF = 1.5


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(device):
    model = PhaseNet(
        feature_dim=128, latent_dim=32, hidden_dim=128, tcn_layers=4,
        encoder_channels=[16, 32, 64, 128], encoder_expand_ratio=4,
        temporal_module="gated_tcn",
    )
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Preprocessing (faithful to the repo's BaseLoader)
# ---------------------------------------------------------------------------
def detect_face_box(frame_rgb):
    detector = cv2.CascadeClassifier(HAAR)
    faces = detector.detectMultiScale(frame_rgb)
    if len(faces) < 1:
        return None
    box = list(faces[int(np.argmax(faces[:, 2]))]) if len(faces) >= 2 else list(faces[0])
    box[0] = max(0, box[0] - (LARGE_BOX_COEF - 1.0) / 2 * box[2])
    box[1] = max(0, box[1] - (LARGE_BOX_COEF - 1.0) / 2 * box[3])
    box[2] = LARGE_BOX_COEF * box[2]
    box[3] = LARGE_BOX_COEF * box[3]
    return np.asarray(box, dtype=int)


def read_and_crop_video(video_file):
    """Stream a video: detect the face on the FIRST frame, then crop every frame
    to 128x128. Streaming keeps memory small (UBFC videos are long)."""
    cap = cv2.VideoCapture(video_file)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read {video_file}")
    first_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    box = detect_face_box(first_rgb)

    def crop(rgb):
        f = rgb
        if box is not None:
            f = f[max(box[1], 0):min(box[1] + box[3], f.shape[0]),
                  max(box[0], 0):min(box[0] + box[2], f.shape[1])]
        return cv2.resize(f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    faces = [crop(first_rgb)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces.append(crop(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return np.asarray(faces, dtype=np.uint8), (box is not None)


def read_wave(bvp_file):
    """UBFC ground_truth.txt: first line is the BVP (reference pulse) signal."""
    with open(bvp_file, "r") as f:
        first = f.read().split("\n")[0]
    return np.asarray([float(x) for x in first.split()])


# ---------------------------------------------------------------------------
# Evaluate one subject
# ---------------------------------------------------------------------------
def eval_subject(subject_dir, model, device):
    vid = os.path.join(subject_dir, "vid.avi")
    gt = os.path.join(subject_dir, "ground_truth.txt")
    if not (os.path.isfile(vid) and os.path.isfile(gt)):
        return None

    faces, found = read_and_crop_video(vid)
    bvps = read_wave(gt)

    n = min(len(faces), len(bvps)) // CLIP_LEN          # whole clips only
    if n == 0:
        return None
    faces = faces[:n * CLIP_LEN]
    bvps = bvps[:n * CLIP_LEN]

    # run the model clip by clip, concatenate the outputs (toolbox style)
    preds = []
    for c in range(n):
        clip = faces[c * CLIP_LEN:(c + 1) * CLIP_LEN]
        x = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32)   # (3,T,H,W), 0-255
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            out, _ = model(x)
        preds.append(out[0].float().cpu().numpy())
    pred_video = np.concatenate(preds)

    # the repo's OWN metric: detrend -> band-pass -> FFT for both signals
    hr_label, hr_pred, snr, macc = calculate_metric_per_video(
        pred_video, bvps, fs=FS, diff_flag=False, use_bandpass=True,
        hr_method="FFT", need_macc=True)

    return {
        "subject": os.path.basename(subject_dir),
        "clips": n,
        "hr_pred": float(hr_pred),
        "hr_ref": float(hr_label),
        "abs_err": float(abs(hr_pred - hr_label)),
        "snr": float(snr),
        "macc": float(macc),
        "face_found": found,
        # kept out of the CSV, used for the JSON waveform viewer
        "_pred_signal": pred_video,
        "_ref_signal": bvps,
    }


# ---------------------------------------------------------------------------
# Evaluate a whole dataset folder
# ---------------------------------------------------------------------------
def eval_dataset(data_path, device, limit=None, delete_after=False):
    subjects = sorted(glob.glob(os.path.join(data_path, "subject*")),
                      key=lambda p: int(re.search(r"subject(\d+)", p).group(1)))
    if not subjects:
        sys.exit(f"No 'subject*' folders found under {data_path}")
    # only keep subjects that actually have BOTH files (others may still be downloading)
    ready = [s for s in subjects
             if os.path.isfile(os.path.join(s, "vid.avi"))
             and os.path.isfile(os.path.join(s, "ground_truth.txt"))]
    missing = len(subjects) - len(ready)
    if missing:
        print(f"(skipping {missing} subject folder(s) without both vid.avi + ground_truth.txt)")
    subjects = ready
    if not subjects:
        sys.exit("No subject folder has BOTH vid.avi and ground_truth.txt yet.")
    if limit:
        subjects = subjects[:limit]

    model = load_model(device)
    print(f"Evaluating {len(subjects)} subject(s) on {device}\n")
    print(f"{'subject':<12}{'ref HR':>8}{'pred HR':>9}{'|err|':>8}{'SNR':>8}{'MACC':>7}")
    print("-" * 52)

    rows = []
    for sd in subjects:
        t0 = time.time()
        try:
            r = eval_subject(sd, model, device)
        except Exception as e:
            print(f"{os.path.basename(sd):<12}  ERROR: {type(e).__name__}: {e}")
            continue
        if r is None:
            print(f"{os.path.basename(sd):<12}  (skipped - missing files or too short)")
            continue
        rows.append(r)
        flag = "" if r["face_found"] else "  <-NO FACE (used full frame)"
        print(f"{r['subject']:<12}{r['hr_ref']:>8.1f}{r['hr_pred']:>9.1f}"
              f"{r['abs_err']:>8.1f}{r['snr']:>8.1f}{r['macc']:>7.2f}"
              f"   [{time.time()-t0:4.1f}s]{flag}")

        # Free disk space immediately: the video is no longer needed once we have
        # its result. Keeps at most ONE video on disk when processing many subjects.
        if delete_after:
            vid = os.path.join(sd, "vid.avi")
            try:
                mb = os.path.getsize(vid) / 1024 ** 2
                os.remove(vid)
                print(f"{'':<12}  deleted vid.avi (+{mb:.0f} MB free)")
            except OSError as e:
                print(f"{'':<12}  could not delete vid.avi: {e}")

    if not rows:
        sys.exit("No subjects evaluated.")

    # ---------------------------------------------------------------- subjects
    # Shape the per-subject records the way the dashboard expects. The signed
    # error is what Bland-Altman needs; abs error is what MAE needs.
    subjects_out, waveforms = [], {}
    for r in rows:
        sid = r["subject"]
        signed = r["hr_pred"] - r["hr_ref"]
        subjects_out.append({
            "id": sid,
            "hr_ref_bpm": r["hr_ref"],
            "hr_pred_bpm": r["hr_pred"],
            "error_bpm": signed,                                  # for Bland-Altman
            "abs_error_bpm": r["abs_err"],
            "mean_hr_bpm": (r["hr_pred"] + r["hr_ref"]) / 2.0,     # BA x-axis
            "snr_db": r["snr"],
            "macc": r["macc"],
            "clips": r["clips"],
            "duration_s": round(r["clips"] * CLIP_LEN / FS, 1),
            "face_detected": r["face_found"],
        })
        try:
            waveforms[sid] = results_export.waveform_payload(
                sid, r["_pred_signal"], r["_ref_signal"], fs=FS)
        except Exception as e:
            print(f"  (waveform export failed for {sid}: {e})")

    metrics = results_export.compute_metrics(
        [{"hr_ref": s["hr_ref_bpm"], "hr_pred": s["hr_pred_bpm"],
          "snr_db": s["snr_db"], "macc": s["macc"]} for s in subjects_out])
    ba = metrics["bland_altman"]

    # ----------------------------------------------------------------- report
    print("\n" + "=" * 60)
    print(f"  RESULTS over {metrics['n_subjects']} subject(s)")
    print("=" * 60)
    print(f"  MAE            : {metrics['mae_bpm']:6.2f} BPM   (paper: 0.15 on full UBFC)")
    print(f"  RMSE           : {metrics['rmse_bpm']:6.2f} BPM")
    print(f"  MAPE           : {metrics['mape_pct']:6.2f} %")
    print(f"  Pearson r      : {metrics['pearson_r']:6.3f}")
    print("  -- clinical agreement (Bland-Altman) --")
    print(f"  bias           : {ba['bias_bpm']:+6.2f} BPM   (systematic over/under-estimate)")
    if ba["loa_lower_bpm"] is not None:
        print(f"  95% limits     : {ba['loa_lower_bpm']:+6.2f} .. {ba['loa_upper_bpm']:+6.2f} BPM")
    print(f"  within 3 BPM   : {metrics['within_3bpm_pct']:5.1f} %")
    print(f"  within 5 BPM   : {metrics['within_5bpm_pct']:5.1f} %")
    print(f"  mean SNR       : {metrics['mean_snr_db']:6.2f} dB")
    print(f"  mean MACC      : {metrics['mean_macc']:6.2f}")
    if metrics["n_subjects"] < 5:
        print("\n  CAUTION: very few subjects - these numbers are indicative, not conclusive.")

    # -------------------------------------------------------------- CSV (raw)
    csv_path = os.path.join(HERE, "ubfc_results.csv")
    csv_cols = [k for k in rows[0].keys() if not k.startswith("_")]
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    # ------------------------------------------------------- JSON (dashboard)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"phasenet_ubfc_{stamp}"
    json_path = results_export.write_results(
        run_id=run_id,
        model={
            "name": "PHASE-Net",
            "variant": "pretrained (authors' UBFC weights)",
            "weights_file": os.path.basename(WEIGHTS),
            "weights_sha256": results_export.file_sha256(WEIGHTS),
            "params_total": 3300274,
            "params_inference_path": 812290,
            "params_claimed_in_paper": 290000,
            "config": {"feature_dim": 128, "latent_dim": 32, "hidden_dim": 128,
                       "tcn_layers": 4, "encoder_channels": [16, 32, 64, 128],
                       "temporal_module": "gated_tcn"},
            "paper": {"title": "PHASE-Net: Physics-Grounded Harmonic Attention System "
                               "for Efficient Remote Photoplethysmography Measurement",
                      "arxiv": "2509.24850", "venue": "CVPR 2026 (Highlight)"},
        },
        dataset={
            "name": "UBFC-rPPG", "subset": "DATASET_2", "path": data_path,
            "fs_hz": FS, "n_subjects_available": len(subjects),
            "ground_truth": "CMS50E pulse oximeter",
        },
        preprocessing={
            "face_detection": "Haar cascade (HC), first frame only",
            "large_box_coef": LARGE_BOX_COEF,
            "resize": [IMG_SIZE, IMG_SIZE],
            "interpolation": "INTER_AREA",
            "pixel_range": "0-255 (no /255 normalisation, matching the toolbox)",
            "clip_length": CLIP_LEN,
            "data_type": "Raw",
        },
        hr_extraction={
            "method": "repo calculate_metric_per_video",
            "steps": ["detrend (lambda=100)",
                      "Butterworth bandpass 0.75-2.5 Hz (45-150 BPM)",
                      "FFT peak"],
            "diff_flag": False,
        },
        subjects=subjects_out,
        waveforms=waveforms,
        device=str(device),
        notes="Inference only - no training. Non-overlapping 128-frame clips, "
              "concatenated per video before HR extraction (toolbox convention).",
    )

    print(f"\n  CSV  : {csv_path}")
    print(f"  JSON : {json_path}")
    print(f"  index: {os.path.join(results_export.RESULTS_DIR, 'index.json')}")
    return metrics["mae_bpm"]


# ---------------------------------------------------------------------------
# Self-test - no dataset needed
# ---------------------------------------------------------------------------
def selftest(device):
    print("=" * 52)
    print("SELF-TEST (no dataset)")
    print("=" * 52)

    # [A] metric wiring: two clean 72-BPM sines -> both HRs ~72, error ~0
    t = np.arange(900) / FS
    sig = np.sin(2 * np.pi * (72.0 / 60.0) * t)
    hr_label, hr_pred, snr, macc = calculate_metric_per_video(
        sig, sig, fs=FS, diff_flag=False, hr_method="FFT")
    print(f"[A] metric on identical 72 BPM sines -> ref {hr_label:.1f}, "
          f"pred {hr_pred:.1f}, err {abs(hr_pred-hr_label):.2f}, MACC {macc:.2f}")
    assert abs(hr_pred - 72) < 2 and abs(hr_label - 72) < 2, "metric wiring broken!"
    print("    PASS")

    # [B] full per-subject path on a fake in-memory 'subject'
    print("\n[B] model path on fake frames:")
    model = load_model(device)
    faces = np.random.randint(0, 255, (300, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    n = len(faces) // CLIP_LEN
    preds = []
    for c in range(n):
        clip = faces[c * CLIP_LEN:(c + 1) * CLIP_LEN]
        x = torch.from_numpy(np.transpose(clip, (3, 0, 1, 2)).astype(np.float32))
        with torch.no_grad():
            out, _ = model(x.unsqueeze(0).to(device))
        preds.append(out[0].float().cpu().numpy())
    pred_video = np.concatenate(preds)
    hr_label, hr_pred, snr, macc = calculate_metric_per_video(
        pred_video, np.random.randn(len(pred_video)), fs=FS, diff_flag=False, hr_method="FFT")
    print(f"    ran {n} clips -> pred HR {hr_pred:.1f} (meaningless on noise). PASS")
    print("\nSELF-TEST COMPLETE.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, help="UBFC root folder containing subject1, subject2, ...")
    ap.add_argument("--limit", type=int, default=None, help="only the first N subjects")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--delete-after", action="store_true",
                    help="delete each vid.avi right after scoring it (saves disk space; "
                         "results are kept in ubfc_results.csv)")
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"Device: {device}")

    if args.selftest:
        selftest(device)
        return
    if not args.data:
        sys.exit("Please pass --data <UBFC folder>  (or --selftest). See the header for the layout.")
    eval_dataset(args.data, device, limit=args.limit, delete_after=args.delete_after)


if __name__ == "__main__":
    main()
