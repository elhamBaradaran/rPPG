"""
Live side-by-side motion test: POS vs PHASE-Net, with the recording kept for later.

THE QUESTION
  Does the deep model actually hold up when the head moves? That is the claim that
  justifies moving away from the classical baseline, and it is the situation KEIKO
  cares about - nobody sits perfectly still while working with a robot.

HOW IT IS MEASURED WITHOUT AN OXIMETER
  The recording is split into three phases:

      phase 1   sit still          <- establishes the true heart rate
      phase 2   move your head     <- the test
      phase 3   sit still again    <- confirms the heart rate did not really change

  Heart rate barely changes over 90 seconds, so the still phases act as the reference
  for the moving phase. If a method drops from 75 to 40 BPM while you move, it has
  failed - no external sensor needed to see that.

WHY THREADS
  A single loop doing capture + display + inference was measured at 10 fps, while the
  camera itself does 30 (see cam_check.py). Capture therefore runs in its own thread
  and does nothing but grab and crop, so frames arrive at the camera's true rate.
  Display and model inference run separately and are allowed to be slower.

USAGE
    python 11_live_motion_test.py
    python 11_live_motion_test.py --still 20 --motion 20     # shorter phases
    python 11_live_motion_test.py --camera 1
Press 'q' to abort.
"""

import argparse
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _common import CLIP_LEN, IMG_SIZE, FS, detect_face_box, load_model  # noqa: E402
from hr_methods import LOW_HZ, HIGH_HZ                                    # noqa: E402
from scipy.signal import butter, filtfilt, periodogram                    # noqa: E402

POS_WINDOW_S = 10          # POS classically uses a 10-second buffer


# ---------------------------------------------------------------------------
# fast heart-rate estimation for the live display
# ---------------------------------------------------------------------------
def bpm_from_signal(sig, fs):
    """Band-pass then take the strongest frequency in the human range."""
    if len(sig) < fs * 4:
        return float("nan")
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean()
    b, a = butter(2, [LOW_HZ / (fs / 2), HIGH_HZ / (fs / 2)], btype="band")
    x = filtfilt(b, a, x)
    f, p = periodogram(x, fs=fs, nfft=1 << 14, detrend=False)
    m = (f >= LOW_HZ) & (f <= HIGH_HZ)
    return float(f[m][np.argmax(p[m])] * 60)


def pos_pulse(rgb):
    """POS projection - the same maths as Models/POS/heartrate.py."""
    x = np.asarray(rgb, dtype=np.float64).T          # (3, T)
    mean = x.mean(axis=1, keepdims=True)
    n = x / np.where(mean == 0, 1e-9, mean)
    s1 = n[1] - n[2]
    s2 = n[1] + n[2] - 2 * n[0]
    alpha = np.std(s1) / (np.std(s2) + 1e-9)
    return s1 + alpha * s2


# ---------------------------------------------------------------------------
# capture thread - grabs and crops, nothing else
# ---------------------------------------------------------------------------
class Capture(threading.Thread):
    def __init__(self, cap, box):
        super().__init__(daemon=True)
        self.cap, self.box = cap, box
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.crops = []              # every 128x128 crop, kept for offline analysis
        self.rgb = []                # spatial mean RGB per frame, for POS
        self.times = []
        self.recent = deque(maxlen=CLIP_LEN)   # last 128 crops, for the model
        self.latest_display = None

    def run(self):
        b = self.box
        while not self.stop_flag.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            f = rgb_full[max(b[1], 0):min(b[1] + b[3], rgb_full.shape[0]),
                         max(b[0], 0):min(b[0] + b[2], rgb_full.shape[1])]
            crop = cv2.resize(f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            with self.lock:
                self.crops.append(crop)
                self.rgb.append(crop.reshape(-1, 3).mean(axis=0))
                self.times.append(time.time())
                self.recent.append(crop)
                self.latest_display = frame

    def snapshot_clip(self):
        with self.lock:
            return list(self.recent) if len(self.recent) == CLIP_LEN else None


# ---------------------------------------------------------------------------
# model thread - one inference every couple of seconds
# ---------------------------------------------------------------------------
class ModelWorker(threading.Thread):
    def __init__(self, cap_thread, model, device, fs_getter):
        super().__init__(daemon=True)
        self.cap_thread, self.model, self.device = cap_thread, model, device
        self.fs_getter = fs_getter
        self.stop_flag = threading.Event()
        self.bpm = float("nan")
        self.history = []            # (timestamp, bpm)

    def run(self):
        while not self.stop_flag.is_set():
            clip = self.cap_thread.snapshot_clip()
            if clip is None:
                time.sleep(0.2)
                continue
            arr = np.transpose(np.asarray(clip), (3, 0, 1, 2)).astype(np.float32)
            x = torch.from_numpy(arr).unsqueeze(0).to(self.device)   # keep 0-255
            with torch.no_grad():
                pred, _ = self.model(x)
            wave = pred[0].float().cpu().numpy()
            bpm = bpm_from_signal(wave, self.fs_getter())
            self.bpm = bpm
            self.history.append((time.time(), bpm))


# ---------------------------------------------------------------------------
def wait_for_face(cap):
    """Detect the face once, as in training, and hold that box for the recording."""
    print("\n[Setup] Looking for your face. Sit comfortably, facing the camera.")
    stable, box, t0 = 0, None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera read failed.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        b = detect_face_box(rgb)
        disp = frame.copy()
        if b is not None:
            stable += 1
            box = b
            cv2.rectangle(disp, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (0, 255, 0), 2)
            if stable >= 10:
                if t0 is None:
                    t0 = time.time()
                left = 3 - int(time.time() - t0)
                if left <= 0:
                    return box
                cv2.putText(disp, f"Starting in {left}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            else:
                cv2.putText(disp, "Face found", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            stable, t0 = 0, None
            cv2.putText(disp, "No face detected", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("Motion test", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit("Aborted.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", type=float, default=30.0, help="seconds per still phase")
    ap.add_argument("--motion", type=float, default=30.0, help="seconds of head motion")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "motion_test.npz"))
    args = ap.parse_args()

    phases = [("SIT STILL", args.still), ("MOVE YOUR HEAD", args.motion),
              ("SIT STILL AGAIN", args.still)]
    total = sum(d for _, d in phases)

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"Device: {device}")
    model = load_model(device)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)   # MSMF measured fastest
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FS)

    box = wait_for_face(cap)

    grabber = Capture(cap, box)
    grabber.start()
    time.sleep(0.5)

    fs_est = [float(FS)]
    worker = ModelWorker(grabber, model, device, lambda: fs_est[0])
    worker.start()

    print(f"\n[Recording] {total:.0f}s total: "
          f"{args.still:.0f}s still, {args.motion:.0f}s motion, {args.still:.0f}s still")
    print("Follow the instruction on screen.\n")

    t_start = time.time()
    pos_hist = []
    try:
        while True:
            el = time.time() - t_start
            if el >= total:
                break

            # which phase are we in?
            acc, label, phase_i, phase_left = 0.0, "", 0, 0.0
            for i, (name, dur) in enumerate(phases):
                if el < acc + dur:
                    label, phase_i, phase_left = name, i, acc + dur - el
                    break
                acc += dur

            with grabber.lock:
                n = len(grabber.times)
                if n >= 2:
                    fs_est[0] = max(n / (grabber.times[-1] - grabber.times[0]), 1.0)
                rgb_recent = np.array(grabber.rgb[-int(POS_WINDOW_S * fs_est[0]):]) \
                    if n > 0 else np.zeros((0, 3))
                disp = None if grabber.latest_display is None else grabber.latest_display.copy()

            # POS is cheap enough to recompute every display frame
            pos_bpm = float("nan")
            if len(rgb_recent) >= fs_est[0] * 5:
                pos_bpm = bpm_from_signal(pos_pulse(rgb_recent), fs_est[0])
                pos_hist.append((time.time(), pos_bpm))

            if disp is not None:
                colour = (0, 200, 255) if phase_i == 1 else (0, 255, 0)
                cv2.rectangle(disp, (0, 0), (640, 60), (0, 0, 0), -1)
                cv2.putText(disp, label, (15, 42), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, colour, 3)
                cv2.putText(disp, f"{phase_left:4.0f}s", (540, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.rectangle(disp, (0, 400), (640, 480), (0, 0, 0), -1)
                cv2.putText(disp, f"POS       {pos_bpm:5.1f} BPM", (15, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 60), 2)
                cv2.putText(disp, f"PHASE-Net {worker.bpm:5.1f} BPM", (15, 462),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 255, 120), 2)
                w = int(636 * el / total)
                cv2.rectangle(disp, (2, 392), (2 + w, 398), (255, 255, 255), -1)
                cv2.putText(disp, f"{fs_est[0]:4.1f} fps", (540, 462),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                cv2.imshow("Motion test", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        grabber.stop_flag.set()
        worker.stop_flag.set()
        time.sleep(0.3)
        cap.release()
        cv2.destroyAllWindows()

    with grabber.lock:
        crops = np.asarray(grabber.crops, dtype=np.uint8)
        times = np.asarray(grabber.times, dtype=np.float64)
    if len(times) < 2:
        sys.exit("Nothing recorded.")

    fps = len(times) / (times[-1] - times[0])
    print(f"[Recording] {len(crops)} frames in {times[-1]-times[0]:.1f}s "
          f"-> {fps:.1f} fps")
    if fps < 25:
        print("  WARNING: below 25 fps - the model was trained at 30, results will suffer.")

    bounds = np.cumsum([0.0] + [d for _, d in phases])
    np.savez_compressed(
        args.out,
        crops=crops,
        times=times - times[0],
        fps=np.array([fps]),
        phase_names=np.array([n for n, _ in phases]),
        phase_bounds=bounds,
        live_pos=np.array([(t - times[0], b) for t, b in pos_hist], dtype=np.float64)
        if pos_hist else np.zeros((0, 2)),
        live_phasenet=np.array([(t - times[0], b) for t, b in worker.history],
                               dtype=np.float64) if worker.history else np.zeros((0, 2)),
    )
    print(f"\nSaved -> {args.out} ({os.path.getsize(args.out)/1024**2:.0f} MB)")
    print("Next:  python 12_motion_analysis.py")


if __name__ == "__main__":
    main()
