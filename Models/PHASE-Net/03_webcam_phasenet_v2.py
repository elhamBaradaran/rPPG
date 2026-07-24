"""
Milestone 2 (v2) - Run PHASE-Net on the webcam with an HONEST heart-rate readout.

WHY v2 EXISTS
  v1 had three real problems, all in the HR-estimation step (the model itself was fine):
    1. It glued independent 128-frame clips end-to-end and ran one FFT. The seams
       injected a fake slow rhythm, so the headline number was the LEAST reliable.
    2. Its FFT could only land on values 14 BPM apart (30 fps / 128 frames), so
       "70 vs 84" was just two neighbouring bins - not real information.
    3. It printed "spread = trustworthy", which was misleading.

WHAT v2 DOES DIFFERENTLY
  * Overlapping sliding windows (default stride 64 = 50% overlap), blended with a
    Hann taper (overlap-add) into ONE continuous waveform - no hard seams.
  * Zero-padded FFT -> ~0.1 BPM resolution instead of 14 BPM.
  * THREE independent trust signals, reported honestly:
       (a) global FFT HR vs peak-counting HR  -> do two different methods agree?
       (b) window-to-window spread             -> is it stable over time?
       (c) SNR in dB                           -> how much does the pulse dominate noise?
  * Saves the waveform (.npy) + a per-window table (.csv) so we can re-analyse
    WITHOUT re-recording. Optionally saves the cropped faces too (--save-frames).

RUN (VS Code terminal, (venv) active):
    python 03_webcam_phasenet_v2.py                 # record 30 s from the webcam
    python 03_webcam_phasenet_v2.py --seconds 45    # longer = better
    python 03_webcam_phasenet_v2.py --selftest      # no webcam; proves the maths
    python 03_webcam_phasenet_v2.py --camera 1      # different webcam
    python 03_webcam_phasenet_v2.py --cpu           # force CPU
"""

import argparse
import csv
import os
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

from neural_methods.model.PhaseNet.PhaseNet import PhaseNet            # noqa: E402
from evaluation.post_process import _detrend, _calculate_SNR          # noqa: E402
from scipy.signal import butter, filtfilt, periodogram, find_peaks    # noqa: E402

CLIP_LEN = 128
IMG_SIZE = 128
TRAIN_FPS = 30
LARGE_BOX_COEF = 1.5
LOW_HZ, HIGH_HZ = 0.75, 2.5          # 45-150 BPM band (same as the repo)


# ===========================================================================
# Model
# ===========================================================================
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


# ===========================================================================
# Preprocessing (faithful to the repo's BaseLoader)
# ===========================================================================
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


def crop_and_resize(frames_rgb, box):
    out = np.empty((len(frames_rgb), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, f in enumerate(frames_rgb):
        if box is not None:
            f = f[max(box[1], 0):min(box[1] + box[3], f.shape[0]),
                  max(box[0], 0):min(box[0] + box[2], f.shape[1])]
        out[i] = cv2.resize(f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return out


def to_model_input(clip_uint8):
    """(T,H,W,3) uint8 -> (1,3,T,H,W) float32, kept in 0-255 (repo uses no /255)."""
    x = np.transpose(clip_uint8, (3, 0, 1, 2)).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0)


# ===========================================================================
# Signal -> heart rate  (the part v1 got wrong)
# ===========================================================================
def clean_signal(sig, fs):
    """Repo-faithful cleaning: detrend (lambda=100) then 0.75-2.5 Hz band-pass."""
    sig = _detrend(np.asarray(sig, dtype=np.float64), 100)
    b, a = butter(1, [LOW_HZ / fs * 2, HIGH_HZ / fs * 2], btype="bandpass")
    return filtfilt(b, a, np.double(sig))


def fft_hr(sig, fs, nfft=2 ** 16):
    """Heart rate via a ZERO-PADDED FFT peak.

    Zero-padding to a large nfft interpolates the spectrum so the peak is located
    to ~0.03 BPM, instead of being quantised to fs/128*60 = 14 BPM as in v1.
    (Note: this refines the PEAK LOCATION; the fundamental uncertainty still comes
    from the recording length, which is why we also record longer.)
    """
    sig = sig - np.mean(sig)
    f, pxx = periodogram(sig, fs=fs, nfft=nfft, detrend=False)
    band = (f >= LOW_HZ) & (f <= HIGH_HZ)
    return float(f[band][np.argmax(pxx[band])] * 60.0)


def peak_hr(sig, fs):
    """Independent cross-check: heart rate from counting waveform peaks in time."""
    min_gap = int(fs * 60.0 / 180.0)                 # no faster than 180 BPM
    prom = 0.3 * np.std(sig)
    peaks, _ = find_peaks(sig, distance=max(min_gap, 1), prominence=prom)
    if len(peaks) < 2:
        return float("nan")
    return float(60.0 / (np.mean(np.diff(peaks)) / fs))


def snr_db(sig, fs, hr_bpm):
    """Signal-to-noise ratio (dB) of the pulse: power at the HR (1st+2nd harmonic)
    vs the rest of the band. High = a clean, dominant pulse. Ground-truth-free."""
    try:
        return float(_calculate_SNR(sig, hr_bpm, fs=fs, low_pass=LOW_HZ, high_pass=HIGH_HZ))
    except Exception:
        return float("nan")


# ===========================================================================
# Sliding-window inference with smooth overlap-add
# ===========================================================================
def infer_waveform(faces, model, device, stride=64):
    """Run the model on overlapping 128-frame windows and blend them into one
    continuous waveform. Returns (continuous_signal, list_of_window_dicts)."""
    T = len(faces)
    starts = list(range(0, max(T - CLIP_LEN, 0) + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] != T - CLIP_LEN and T >= CLIP_LEN:
        starts.append(T - CLIP_LEN)                  # make sure the tail is covered

    hann = np.hanning(CLIP_LEN) + 1e-6               # taper down window edges
    cont = np.zeros(T, dtype=np.float64)
    wsum = np.zeros(T, dtype=np.float64)
    windows = []

    for s in starts:
        clip = faces[s:s + CLIP_LEN]
        with torch.no_grad():
            pred, _ = model(to_model_input(clip).to(device))
        w = pred[0].float().cpu().numpy()
        # standardise each window so magnitudes match before blending
        w = (w - w.mean()) / (w.std() + 1e-8)
        cont[s:s + CLIP_LEN] += w * hann
        wsum[s:s + CLIP_LEN] += hann
        windows.append({"start": s, "raw": w})

    cont /= np.maximum(wsum, 1e-6)
    return cont, windows


# ===========================================================================
# Analyse + report
# ===========================================================================
def analyse(faces, fps, model, device, tag="webcam", save_frames=False):
    T = len(faces)
    if T < CLIP_LEN:
        sys.exit(f"Need at least {CLIP_LEN} frames, got {T}.")

    print(f"\n[Model] {T} frames ({T / fps:.1f}s) on {device}, sliding 128-frame windows...")
    cont, windows = infer_waveform(faces, model, device)

    # --- global estimate on the full continuous waveform ---
    filt = clean_signal(cont, fps)
    g_fft = fft_hr(filt, fps)
    g_peak = peak_hr(filt, fps)
    g_snr = snr_db(filt, fps, g_fft)

    # --- per-window estimates (do NOT rely on gluing) ---
    per_fft, per_peak = [], []
    for w in windows:
        fw = clean_signal(w["raw"], fps)
        per_fft.append(fft_hr(fw, fps))
        per_peak.append(peak_hr(fw, fps))
    per_fft = np.array(per_fft)
    per_peak_valid = np.array([p for p in per_peak if not np.isnan(p)])

    med_fft = float(np.median(per_fft))
    p25, p75 = np.percentile(per_fft, [25, 75])
    iqr = float(p75 - p25)
    med_peak = float(np.median(per_peak_valid)) if per_peak_valid.size else float("nan")

    # ------------------------------------------------------------------ report
    print("\n" + "=" * 64)
    print(f"  BEST ESTIMATE (median over {len(windows)} windows):  {med_fft:.1f} BPM")
    print("=" * 64)
    print("  Cross-checks (independent methods should agree):")
    print(f"    global FFT (one long signal) : {g_fft:6.1f} BPM")
    print(f"    global peak-count            : {g_peak:6.1f} BPM")
    print(f"    per-window peak-count median : {med_peak:6.1f} BPM")
    print("  Trust signals:")
    print(f"    window-to-window spread(IQR) : {iqr:6.1f} BPM   (smaller = steadier)")
    print(f"    FFT vs peak agreement        : {abs(g_fft - g_peak):6.1f} BPM apart")
    print(f"    SNR                          : {g_snr:6.1f} dB    (higher = cleaner pulse)")

    # honest verdict from the three signals
    agree = abs(g_fft - g_peak)
    if iqr <= 6 and agree <= 8 and g_snr > 0:
        verdict = "GOOD - three methods agree and the pulse is clean."
    elif iqr <= 12 and agree <= 15:
        verdict = "FAIR - roughly consistent; treat as approximate."
    else:
        verdict = "UNRELIABLE - methods disagree (likely motion/lighting). Re-record."
    print(f"\n  VERDICT: {verdict}")
    if abs(fps - TRAIN_FPS) > 4:
        print(f"  NOTE: webcam ran at {fps:.1f} fps but the model was trained at "
              f"{TRAIN_FPS} - accuracy may suffer.")

    # ------------------------------------------------------------------ save
    np.save(os.path.join(HERE, f"waveform_{tag}.npy"), cont)
    with open(os.path.join(HERE, f"windows_{tag}.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["window", "start_frame", "fft_bpm", "peak_bpm"])
        for i, w in enumerate(windows):
            wr.writerow([i, w["start"], round(per_fft[i], 2),
                         "" if np.isnan(per_peak[i]) else round(per_peak[i], 2)])
    if save_frames:
        np.save(os.path.join(HERE, f"faces_{tag}.npy"), faces)
        print(f"  (saved cropped faces -> faces_{tag}.npy)")

    _save_plot(cont, filt, fps, med_fft, g_snr, tag)
    print(f"  Saved: waveform_{tag}.npy, windows_{tag}.csv, waveform_{tag}.png")
    return med_fft


def _save_plot(cont, filt, fps, bpm, snr, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(len(filt)) / fps
        fig, ax = plt.subplots(2, 1, figsize=(11, 6))
        ax[0].plot(t, cont, lw=0.9)
        ax[0].set_title("PHASE-Net continuous output (overlap-add)")
        ax[0].set_xlabel("seconds")
        ax[1].plot(t, filt, lw=1.1, color="crimson")
        ax[1].set_title(f"Cleaned pulse  ->  {bpm:.1f} BPM   (SNR {snr:.1f} dB)")
        ax[1].set_xlabel("seconds")
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"waveform_{tag}.png"), dpi=130)
    except Exception as e:
        print(f"  (plot skipped: {e})")


# ===========================================================================
# Webcam capture (same proven logic as v1, just longer by default)
# ===========================================================================
def capture_frames(camera, n_frames):
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera}. Try --camera 1")
    # Force MJPG. Many webcams only reach 30 fps at 640x480 in compressed MJPG
    # mode; the default uncompressed (YUY2) mode is USB-bandwidth-limited to
    # ~10 fps - which corrupts the timing the model depends on.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, TRAIN_FPS)
    print(f"[Camera] driver reports "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} fps requested")

    print("\n[Preview] Recording starts when your face is detected. Sit still, "
          "steady light. Press 'q' to abort.\n")
    stable, box, t_start = 0, None, None
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            cap.release(); raise RuntimeError("Camera read failed.")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        b = detect_face_box(rgb)
        disp = frame_bgr.copy()
        if b is not None:
            stable += 1; box = b
            cv2.rectangle(disp, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (0, 255, 0), 2)
            if stable >= 10:
                if t_start is None:
                    t_start = time.time()
                left = 3 - int(time.time() - t_start)
                if left <= 0:
                    break
                cv2.putText(disp, f"Starting in {left}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            else:
                cv2.putText(disp, "Face found - hold still", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            stable, t_start = 0, None
            cv2.putText(disp, "No face detected", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("PHASE-Net v2", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release(); cv2.destroyAllWindows(); sys.exit("Aborted.")

    print(f"[Recording] Capturing {n_frames} frames (~{n_frames / TRAIN_FPS:.0f}s)...")
    frames, t0 = [], time.time()
    while len(frames) < n_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        disp = frame_bgr.copy()
        cv2.rectangle(disp, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        pct = int(100 * len(frames) / n_frames)
        cv2.rectangle(disp, (20, 440), (20 + int(pct * 6), 460), (0, 255, 0), -1)
        cv2.putText(disp, f"REC {pct}%", (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("PHASE-Net v2", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    elapsed = time.time() - t0
    cap.release(); cv2.destroyAllWindows()
    fps = len(frames) / elapsed if elapsed > 0 else TRAIN_FPS
    print(f"[Recording] {len(frames)} frames in {elapsed:.1f}s -> {fps:.1f} fps")
    return frames, fps, box


# ===========================================================================
# Self-test - proves the maths WITHOUT a webcam
# ===========================================================================
def selftest(device):
    print("=" * 64)
    print("SELF-TEST (no webcam)")
    print("=" * 64)

    fs = 30.0
    # A clean pulse at a NON-bin frequency v1 could never have resolved:
    true_bpm = 72.5
    t = np.arange(int(40 * fs)) / fs           # 40 s
    sig = np.sin(2 * np.pi * (true_bpm / 60.0) * t) + 0.15 * np.random.randn(len(t))
    filt = clean_signal(sig, fs)
    f_hr, p_hr = fft_hr(filt, fs), peak_hr(filt, fs)
    print(f"[A] synthetic {true_bpm} BPM -> FFT {f_hr:.2f}, peak {p_hr:.2f}")
    print(f"    v1's 14-BPM grid could only say 70.3 or 84.4; v2 resolves it.")
    assert abs(f_hr - true_bpm) < 1.5 and abs(p_hr - true_bpm) < 3, "HR maths off!"
    print("    PASS")

    print("\n[B] full model pipeline on random frames (values meaningless):")
    model = load_model(device)
    fake = np.random.randint(0, 255, (400, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    analyse(fake, 30.0, model, device, tag="selftest")
    print("    PASS - ran end to end.")
    print("\nSELF-TEST COMPLETE.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--save-frames", action="store_true",
                    help="also save the cropped faces for full offline re-analysis")
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"Device: {device}")

    if args.selftest:
        selftest(device)
        return

    n = max(CLIP_LEN + args.stride, int(args.seconds * TRAIN_FPS))
    frames, fps, box = capture_frames(args.camera, n)
    faces = crop_and_resize(frames, box)
    model = load_model(device)
    analyse(faces, fps, model, device, tag="webcam", save_frames=args.save_frames)


if __name__ == "__main__":
    main()
