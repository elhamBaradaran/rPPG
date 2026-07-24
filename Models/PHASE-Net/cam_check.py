"""
Camera diagnostic - find out WHY the webcam runs slow.

It measures the REAL frame rate (not the driver's claim) under several settings,
so we know whether the bottleneck is the camera, the lighting/exposure, or the
display loop. No face or model needed - it just prints numbers.

RUN:
    python cam_check.py

Do it TWICE: once in your normal light, once with a bright lamp / window on your
face, and compare the "measured" fps.
"""

import time
import cv2

TARGET_W, TARGET_H = 640, 480
N_WARMUP = 15
N_MEASURE = 90            # ~3 s at 30 fps


def measure(cap, show=False, label=""):
    # warm up (let auto-exposure settle)
    for _ in range(N_WARMUP):
        cap.read()
    t0 = time.time()
    got = 0
    for _ in range(N_MEASURE):
        ok, frame = cap.read()
        if not ok:
            break
        got += 1
        if show:
            cv2.imshow("cam_check", frame)
            cv2.waitKey(1)
    dt = time.time() - t0
    fps = got / dt if dt > 0 else 0
    reported = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {label:38s} measured {fps:5.1f} fps   "
          f"(driver claims {reported:4.0f}, {w}x{h})")
    return fps


def open_cam(backend):
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main():
    print("=" * 66)
    print("CAMERA DIAGNOSTIC  (target: 30 fps at 640x480)")
    print("=" * 66)

    backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]

    for name, backend in backends:
        print(f"\nBackend: {name}")
        cap = open_cam(backend)
        if cap is None:
            print("  (could not open)")
            continue

        # 1) plain read, auto-exposure as-is
        measure(cap, show=False, label="auto-exposure, no display")

        # 2) plain read WITH display (is imshow the bottleneck?)
        measure(cap, show=True, label="auto-exposure, WITH display")

        # 3) force MANUAL short exposure -> should raise fps if light was the cause
        #    (image gets darker; that is the trade-off)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual on many drivers
        for exp in (-4, -6):
            cap.set(cv2.CAP_PROP_EXPOSURE, exp)
            measure(cap, show=False, label=f"manual exposure={exp}, no display")

        cap.release()

    cv2.destroyAllWindows()
    print("\n" + "=" * 66)
    print("HOW TO READ THIS:")
    print("  * If 'auto-exposure' is ~30  -> your camera is fine; run the demo again.")
    print("  * If 'auto-exposure' is ~10 but 'manual exposure' is ~30")
    print("      -> LIGHT is the problem. Add a bright lamp/window and re-run the demo.")
    print("  * If 'no display' is 30 but 'WITH display' is 10")
    print("      -> the display loop is the bottleneck (I will fix the script).")
    print("  * If EVERYTHING is ~10 -> the camera hardware caps here; we adapt in software.")
    print("=" * 66)


if __name__ == "__main__":
    main()
