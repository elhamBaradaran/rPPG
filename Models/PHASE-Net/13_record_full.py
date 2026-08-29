"""
Record a motion test while keeping enough of the frame for the face to move within.

WHY THIS EXISTS
  In the first motion test PHASE-Net destabilised under head movement while POS did not.
  The suspected cause is not the model but the CROP: face detection runs on the first
  frame only (matching the training config), so when the head moves the face leaves that
  fixed box and never returns to exactly the same place.

  To test that properly we need to process ONE recording TWO ways - fixed crop versus
  dynamic tracking - so that the crop strategy is the only variable. That needs more than
  the 128x128 crop, but storing whole frames would cost 2.5 GB for 90 seconds, and video
  compression is not an option: lossy codecs destroy the very colour micro-variations the
  pulse lives in.

THE COMPROMISE
  Save a region twice the size of the face box, stored at 256x256:

      +-------------- saved region, 256x256 --------------+
      |                                                   |
      |            +--- face box, 128x128 ---+            |
      |            |                         |            |
      |            |    (the training crop)  |            |
      |            +-------------------------+            |
      |                                                   |
      +---------------------------------------------------+

  Because the region is exactly twice the box, the face occupies exactly 128 px. The
  fixed crop taken from this recording is therefore pixel-identical to what the model saw
  before, with no resampling, while leaving half a face-width of room in every direction
  for the head to move into. About 530 MB for 90 s, uncompressed uint8.

Capture runs in its own thread; a single-threaded loop was measured at 10 fps on hardware
that comfortably delivers 30.

USAGE
    python 13_record_full.py
    python 13_record_full.py --still 20 --motion 20      # shorter
    python 13_record_full.py --region-scale 2.5          # more room to move
Press q to abort.
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _common import FS, detect_face_box   # noqa: E402

REGION_PX = 256          # stored size of the saved region


def region_from_box(box, frame_w, frame_h, scale):
    """A window `scale` times the face box, centred on it and clipped to the frame."""
    x, y, w, h = [float(v) for v in box]
    cx, cy = x + w / 2, y + h / 2
    rw, rh = w * scale, h * scale
    x0 = int(max(0, round(cx - rw / 2)))
    y0 = int(max(0, round(cy - rh / 2)))
    x1 = int(min(frame_w, round(cx + rw / 2)))
    y1 = int(min(frame_h, round(cy + rh / 2)))
    return x0, y0, x1, y1


class Capture(threading.Thread):
    """Grabs frames and crops the saved region - and nothing else, so it keeps up."""

    def __init__(self, cap, rect):
        super().__init__(daemon=True)
        self.cap = cap
        self.x0, self.y0, self.x1, self.y1 = rect
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.regions, self.times = [], []
        self.latest_display = None

    def run(self):
        while not self.stop_flag.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            reg = rgb[self.y0:self.y1, self.x0:self.x1]
            reg = cv2.resize(reg, (REGION_PX, REGION_PX), interpolation=cv2.INTER_AREA)
            with self.lock:
                self.regions.append(reg)
                self.times.append(time.time())
                self.latest_display = frame


def wait_for_face(cap):
    print("\n[Setup] Looking for your face. Sit where you will stay for the recording.")
    stable, box, t0 = 0, None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera read failed.")
        b = detect_face_box(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        disp = frame.copy()
        if b is not None:
            stable, box = stable + 1, b
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
        cv2.imshow("Recording", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit("Aborted.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", type=float, default=30.0)
    ap.add_argument("--motion", type=float, default=30.0)
    ap.add_argument("--region-scale", type=float, default=2.0,
                    help="saved region size as a multiple of the face box")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--label", default=None,
                    help="condition name; the recording is saved as motion_<label>.npz "
                         "and the on-screen instruction matches it "
                         "(still / slow / fast / talk / lean)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # What the middle phase asks for. Each recording is a still/condition/still sandwich,
    # so every condition carries its own baseline and heart-rate drift between recordings
    # cannot contaminate the comparison.
    INSTRUCTIONS = {
        "still": ("STAY STILL (control)", "Do nothing - this measures the noise floor."),
        "slow": ("TURN HEAD SLOWLY", "Left and right, about one full cycle every 4 s."),
        "fast": ("TURN HEAD QUICKLY", "Same movement, about one cycle every 1.5 s."),
        "talk": ("TALK", "Speak continuously but keep your head still."),
        "lean": ("LEAN IN AND OUT", "Move closer to the camera and back again."),
    }
    label = args.label or "full"
    mid_label, mid_hint = INSTRUCTIONS.get(label, ("MOVE YOUR HEAD",
                                                   "Turn left and right, nod, lean."))
    out_path = args.out or os.path.join(HERE, f"motion_{label}.npz")

    phases = [("SIT STILL", args.still),
              (mid_label, args.motion),
              ("SIT STILL AGAIN", args.still)]
    total = sum(d for _, d in phases)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FS)

    box = wait_for_face(cap)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    rect = region_from_box(box, fw, fh, args.region_scale)
    x0, y0, x1, y1 = rect

    # Where the original face box sits inside the stored 256x256 region. Recording this
    # explicitly means the fixed crop can be reproduced exactly even when the region had
    # to be clipped at a frame edge.
    sx = REGION_PX / max(x1 - x0, 1)
    sy = REGION_PX / max(y1 - y0, 1)
    box_in_region = np.array([(box[0] - x0) * sx, (box[1] - y0) * sy,
                              box[2] * sx, box[3] * sy], dtype=np.float64)

    print(f"[Setup] face box {list(map(int, box))} in the frame")
    print(f"        saved region {x1-x0}x{y1-y0} px -> stored at {REGION_PX}x{REGION_PX}")
    print(f"        face occupies {box_in_region[2]:.0f}x{box_in_region[3]:.0f} px of it")

    grabber = Capture(cap, rect)
    grabber.start()
    time.sleep(0.4)

    est_mb = total * FS * REGION_PX * REGION_PX * 3 / 1024 ** 2
    print(f"\n[Recording] condition '{label}', {total:.0f}s, about {est_mb:.0f} MB")
    print(f"  Middle phase: {mid_hint}")
    print("  Keep facing the camera throughout.\n")

    t_start = time.time()
    try:
        while True:
            el = time.time() - t_start
            if el >= total:
                break
            acc, caption, idx, left = 0.0, "", 0, 0.0
            for i, (name, dur) in enumerate(phases):
                if el < acc + dur:
                    caption, idx, left = name, i, acc + dur - el
                    break
                acc += dur

            with grabber.lock:
                n = len(grabber.times)
                fps = (n / (grabber.times[-1] - grabber.times[0])) if n > 2 else 0.0
                disp = None if grabber.latest_display is None else grabber.latest_display.copy()

            if disp is not None:
                colour = (0, 200, 255) if idx == 1 else (0, 255, 0)
                cv2.rectangle(disp, (x0, y0), (x1, y1), (200, 200, 200), 1)
                cv2.rectangle(disp, (box[0], box[1]),
                              (box[0] + box[2], box[1] + box[3]), colour, 2)
                cv2.rectangle(disp, (0, 0), (640, 60), (0, 0, 0), -1)
                cv2.putText(disp, caption, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3)
                cv2.putText(disp, f"{left:4.0f}s", (540, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                w = int(636 * el / total)
                cv2.rectangle(disp, (2, 470), (2 + w, 476), (255, 255, 255), -1)
                cv2.putText(disp, f"{n} frames  {fps:4.1f} fps", (15, 462),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                cv2.imshow("Recording", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        grabber.stop_flag.set()
        time.sleep(0.3)
        cap.release()
        cv2.destroyAllWindows()

    with grabber.lock:
        regions = np.asarray(grabber.regions, dtype=np.uint8)
        times = np.asarray(grabber.times, dtype=np.float64)
    if len(times) < 2:
        sys.exit("Nothing recorded.")

    fps = len(times) / (times[-1] - times[0])
    print(f"[Recording] {len(regions)} frames in {times[-1]-times[0]:.1f}s -> {fps:.1f} fps")
    if fps < 25:
        print("  WARNING: below 25 fps; the model was trained at 30.")

    np.savez_compressed(
        out_path,
        regions=regions,
        label=np.array(label),
        times=times - times[0],
        fps=np.array([fps]),
        box_in_region=box_in_region,
        region_scale=np.array([args.region_scale]),
        phase_names=np.array([n for n, _ in phases]),
        phase_bounds=np.cumsum([0.0] + [d for _, d in phases]),
    )
    print(f"\nSaved -> {out_path} ({os.path.getsize(out_path)/1024**2:.0f} MB)")
    print("Record the next condition, or when all are done:")
    print("  python 15_motion_protocol.py --watch <your smartwatch BPM>")


if __name__ == "__main__":
    main()
