import cv2
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERROR: Could not open the webcam.")
    exit()

FPS = 30
WINDOW_SECONDS = 10
BUFFER_SIZE = FPS * WINDOW_SECONDS

r_buffer = deque(maxlen=BUFFER_SIZE)
g_buffer = deque(maxlen=BUFFER_SIZE)
b_buffer = deque(maxlen=BUFFER_SIZE)

LOW_HZ = 0.7
HIGH_HZ = 4.0

# Three skin regions, each defined as fractions of the face box:
# (left_fraction, top_fraction, width_fraction, height_fraction).
# You can nudge these numbers if a box does not sit nicely on the skin.
REGIONS = [
    (0.25, 0.10, 0.50, 0.20),   # forehead
    (0.13, 0.55, 0.20, 0.17),   # left cheek  (viewer's left)
    (0.67, 0.55, 0.20, 0.17),   # right cheek (viewer's right)
]


def pos_algorithm(r, g, b):
    rgb = np.array([r, g, b])
    mean_rgb = np.mean(rgb, axis=1, keepdims=True)
    rgb_normalized = rgb / mean_rgb
    s1 = rgb_normalized[1] - rgb_normalized[2]
    s2 = rgb_normalized[1] + rgb_normalized[2] - 2 * rgb_normalized[0]
    alpha = np.std(s1) / (np.std(s2) + 1e-9)
    pulse = s1 + alpha * s2
    pulse = pulse - np.mean(pulse)
    return pulse


def bandpass_filter(signal, fps, low, high):
    nyquist = 0.5 * fps
    low_cut = low / nyquist
    high_cut = high / nyquist
    b, a = butter(3, [low_cut, high_cut], btype='band')
    return filtfilt(b, a, signal)


def estimate_bpm(pulse, fps):
    n = len(pulse)
    fft = np.abs(np.fft.rfft(pulse))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    valid = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    if not np.any(valid):
        return 0.0
    peak_freq = freqs[valid][np.argmax(fft[valid])]
    return peak_freq * 60.0


print("Measuring heart rate (forehead + cheeks). Press 'q' to quit.")

bpm_display = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # Measure the average color of each region, then combine them.
        region_means = []
        for (rfx, rfy, rfw, rfh) in REGIONS:
            rx = x + int(w * rfx)
            ry = y + int(h * rfy)
            rw = int(w * rfw)
            rh = int(h * rfh)
            sub = frame[ry:ry + rh, rx:rx + rw]
            if sub.size > 0:
                region_means.append(np.mean(sub, axis=(0, 1)))
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                              (0, 255, 0), 2)

        if len(region_means) > 0:
            # Equal weight per region, so the cheeks contribute as much
            # as the forehead (not drowned out by the larger forehead).
            avg_color = np.mean(region_means, axis=0)
            b_buffer.append(avg_color[0])
            g_buffer.append(avg_color[1])
            r_buffer.append(avg_color[2])

    wave_img = np.zeros((200, BUFFER_SIZE, 3), dtype=np.uint8)

    if len(g_buffer) == BUFFER_SIZE:
        pulse = pos_algorithm(
            np.array(r_buffer), np.array(g_buffer), np.array(b_buffer)
        )
        filtered = bandpass_filter(pulse, FPS, LOW_HZ, HIGH_HZ)
        bpm_display = estimate_bpm(filtered, FPS)

        p_min, p_max = np.min(filtered), np.max(filtered)
        if p_max - p_min > 1e-9:
            pulse_scaled = (filtered - p_min) / (p_max - p_min) * 180 + 10
            for i in range(1, len(pulse_scaled)):
                x1, y1 = i - 1, int(200 - pulse_scaled[i - 1])
                x2, y2 = i,     int(200 - pulse_scaled[i])
                cv2.line(wave_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.putText(wave_img, "Filtered pulse (multi-ROI)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    else:
        msg = f"Collecting... {len(g_buffer)} / {BUFFER_SIZE}"
        cv2.putText(wave_img, msg, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    bpm_text = f"Heart Rate: {bpm_display:.1f} BPM"
    cv2.putText(frame, bpm_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Camera - press q to quit", frame)
    cv2.imshow("Pulse Waveform", wave_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Closed.")