"""
Milestone 2 - Run PHASE-Net on a REAL face from your webcam.

WHAT THIS DOES
  1. Opens your webcam and waits until it finds your face.
  2. Records a few seconds of video.
  3. Preprocesses the frames EXACTLY the way PHASE-Net was trained
     (Haar-cascade face box -> enlarged 1.5x -> resized to 128x128 -> RGB, 0-255).
  4. Feeds 128-frame clips to PHASE-Net.
  5. Converts the predicted pulse waveform into a heart rate (BPM),
     using the repo's own post-processing (detrend -> bandpass -> FFT).
  6. Saves a plot of your pulse waveform.

HOW TO RUN (in the VS Code terminal, with (venv) active):
    python 02_webcam_phasenet.py

Options:
    python 02_webcam_phasenet.py --seconds 20      # record longer (more accurate)
    python 02_webcam_phasenet.py --camera 1        # use a different webcam
    python 02_webcam_phasenet.py --selftest        # no webcam; just check the code works

Press 'q' at any time to abort.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Point Python at the PhaseNet repo so we can import the model + post-processing
# ---------------------------------------------------------------------------
REPO = r"D:\00-TU-CLAUSTHAL\keiko-rppg\PhaseNet"
sys.path.insert(0, REPO)

WEIGHTS = os.path.join(REPO, "weights", "phasenet_ubfc_epoch9.pth")
HAAR = os.path.join(REPO, "dataset", "haarcascade_frontalface_default.xml")

from neural_methods.model.PhaseNet.PhaseNet import PhaseNet          # noqa: E402
from evaluation.post_process import _detrend, _calculate_fft_hr      # noqa: E402
from scipy.signal import butter, filtfilt                            # noqa: E402

# These MUST match how the model was trained (from the repo's config YAML)
CLIP_LEN = 128       # frames per clip
IMG_SIZE = 128       # 128 x 128 pixels
TRAIN_FPS = 30       # the model was trained on 30 fps video
LARGE_BOX_COEF = 1.5


# ===========================================================================
# Model
# ===========================================================================
def load_model(device):
    """Build PHASE-Net and load the authors' UBFC-trained weights."""
    model = PhaseNet(
        feature_dim=128, latent_dim=32, hidden_dim=128, tcn_layers=4,
        encoder_channels=[16, 32, 64, 128], encoder_expand_ratio=4,
        temporal_module="gated_tcn",
    )
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  WARNING: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    return model


# ===========================================================================
# Preprocessing - copied faithfully from the repo's BaseLoader
# ===========================================================================
def detect_face_box(frame_rgb):
    """Haar-cascade face detection, then enlarge the box by 1.5x (as in training).

    Returns [x, y, w, h]. Falls back to the whole frame if no face is found.
    """
    detector = cv2.CascadeClassifier(HAAR)
    faces = detector.detectMultiScale(frame_rgb)
    if len(faces) < 1:
        return None
    # If several faces, keep the biggest one (same rule as the repo)
    box = list(faces[int(np.argmax(faces[:, 2]))]) if len(faces) >= 2 else list(faces[0])
    # Enlarge around the centre
    box[0] = max(0, box[0] - (LARGE_BOX_COEF - 1.0) / 2 * box[2])
    box[1] = max(0, box[1] - (LARGE_BOX_COEF - 1.0) / 2 * box[3])
    box[2] = LARGE_BOX_COEF * box[2]
    box[3] = LARGE_BOX_COEF * box[3]
    return np.asarray(box, dtype=int)


def crop_and_resize(frames_rgb, box):
    """Crop every frame with the SAME box (training used static detection) and resize."""
    out = np.empty((len(frames_rgb), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, f in enumerate(frames_rgb):
        if box is not None:
            f = f[max(box[1], 0):min(box[1] + box[3], f.shape[0]),
                  max(box[0], 0):min(box[0] + box[2], f.shape[1])]
        out[i] = cv2.resize(f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return out


def to_model_input(clip_uint8):
    """(T,H,W,3) uint8  ->  (1,3,T,H,W) float32 in the 0-255 range.

    IMPORTANT: the repo does np.float32(data) with NO /255, so we must not
    normalise here either, or the model sees the wrong input scale.
    """
    x = np.transpose(clip_uint8, (3, 0, 1, 2))       # -> (3, T, H, W)
    x = np.float32(x)                                 # keep 0-255!
    return torch.from_numpy(x).unsqueeze(0)           # -> (1, 3, T, H, W)


# ===========================================================================
# Heart rate - copied from the repo's calculate_metric_per_video (diff_flag=False)
# ===========================================================================
def waveform_to_bpm(sig, fs):
    """detrend -> bandpass 0.75-2.5 Hz (45-150 BPM) -> FFT peak -> BPM."""
    sig = _detrend(np.asarray(sig, dtype=np.float64), 100)
    b, a = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype="bandpass")
    sig = filtfilt(b, a, np.double(sig))
    return float(_calculate_fft_hr(sig, fs=fs)), sig


# ===========================================================================
# Webcam capture
# ===========================================================================
def capture_frames(camera, n_frames):
    """Show a preview, wait for a face, then record n_frames. Returns (frames_rgb, fps)."""
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera}. Try --camera 1")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, TRAIN_FPS)

    print("\n[Preview] Look at the camera. Recording starts when your face is detected.")
    print("          Sit still, keep lighting steady. Press 'q' to abort.\n")

    # --- Phase 1: wait for a stable face, with a short countdown -----------
    stable, box, t_start = 0, None, None
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Failed to read from the camera.")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        b = detect_face_box(rgb)
        disp = frame_bgr.copy()

        if b is not None:
            stable += 1
            box = b
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

        cv2.imshow("PHASE-Net - webcam", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release(); cv2.destroyAllWindows()
            sys.exit("Aborted by user.")

    # --- Phase 2: record ---------------------------------------------------
    print(f"[Recording] Capturing {n_frames} frames...")
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
        cv2.imshow("PHASE-Net - webcam", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = time.time() - t0
    cap.release()
    cv2.destroyAllWindows()

    fps = len(frames) / elapsed if elapsed > 0 else TRAIN_FPS
    print(f"[Recording] Got {len(frames)} frames in {elapsed:.1f}s -> measured {fps:.1f} fps")
    return frames, fps, box


# ===========================================================================
# Main
# ===========================================================================
def run(frames_rgb, fps, box, device, save_plot=True):
    model = load_model(device)

    faces = crop_and_resize(frames_rgb, box)
    n_clips = len(faces) // CLIP_LEN
    if n_clips == 0:
        sys.exit(f"Not enough frames: need at least {CLIP_LEN}, got {len(faces)}.")
    print(f"\n[Model] Processing {n_clips} clip(s) of {CLIP_LEN} frames on {device}...")

    pieces, per_clip = [], []
    for c in range(n_clips):
        clip = faces[c * CLIP_LEN:(c + 1) * CLIP_LEN]
        x = to_model_input(clip).to(device)
        with torch.no_grad():
            pred, _ = model(x)
        w = pred[0].float().cpu().numpy()
        pieces.append(w)
        bpm_c, _ = waveform_to_bpm(w, fps)
        per_clip.append(bpm_c)
        print(f"   clip {c + 1}/{n_clips}:  {bpm_c:6.2f} BPM")

    full = np.concatenate(pieces)
    bpm, filtered = waveform_to_bpm(full, fps)

    print("\n" + "=" * 62)
    print(f"  ESTIMATED HEART RATE:  {bpm:.1f} BPM")
    print("=" * 62)
    print(f"  per-clip values : {[round(v, 1) for v in per_clip]}")
    print(f"  spread          : {max(per_clip) - min(per_clip):.1f} BPM "
          f"(small = stable/trustworthy)")
    print(f"  video used      : {len(full)} frames at {fps:.1f} fps "
          f"= {len(full) / fps:.1f} s")
    if abs(fps - TRAIN_FPS) > 4:
        print(f"  NOTE: your webcam ran at {fps:.1f} fps but the model was trained "
              f"at {TRAIN_FPS} fps - accuracy may suffer.")

    if save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            t = np.arange(len(filtered)) / fps
            fig, ax = plt.subplots(2, 1, figsize=(11, 6))
            ax[0].plot(t, full, lw=0.9)
            ax[0].set_title("PHASE-Net raw output")
            ax[0].set_xlabel("seconds")
            ax[1].plot(t, filtered, lw=1.1, color="crimson")
            ax[1].set_title(f"After detrend + bandpass  ->  {bpm:.1f} BPM")
            ax[1].set_xlabel("seconds")
            for a in ax:
                a.grid(alpha=0.3)
            fig.tight_layout()
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pulse_waveform.png")
            fig.savefig(out, dpi=130)
            print(f"\n  Waveform plot saved to: {out}")
        except Exception as e:
            print(f"  (plot skipped: {e})")

    return bpm


def selftest(device):
    """Check the code paths without a webcam."""
    print("=" * 62)
    print("SELF-TEST (no webcam)")
    print("=" * 62)

    # 1) Does the BPM maths work on a known signal? 75 BPM = 1.25 Hz
    fs = 30.0
    t = np.arange(600) / fs
    sine = np.sin(2 * np.pi * 1.25 * t)
    bpm, _ = waveform_to_bpm(sine, fs)
    print(f"[A] BPM maths on a synthetic 75.0 BPM sine -> {bpm:.2f} BPM")
    assert 72 < bpm < 78, "BPM maths is wrong!"
    print("    PASS")

    # 2) Does the full model path run end to end on fake frames?
    fake = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(CLIP_LEN)]
    print("\n[B] Full pipeline on random frames (values will be meaningless):")
    run(fake, 30.0, None, device, save_plot=False)
    print("    PASS - no crashes, shapes all line up.")
    print("\nSELF-TEST COMPLETE.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=13.0, help="how long to record")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--selftest", action="store_true", help="run without a webcam")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"Device: {device}")

    if args.selftest:
        selftest(device)
        return

    n_frames = int(args.seconds * TRAIN_FPS)
    n_frames = max(CLIP_LEN, (n_frames // CLIP_LEN) * CLIP_LEN)  # whole clips only
    frames, fps, box = capture_frames(args.camera, n_frames)
    run(frames, fps, box, device)


if __name__ == "__main__":
    main()
