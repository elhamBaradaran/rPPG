import cv2
import numpy as np
from collections import deque

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERROR: Could not open the webcam.")
    exit()

FPS = 30
WINDOW_SECONDS = 10
BUFFER_SIZE = FPS * WINDOW_SECONDS      # 300 frames = ~10 seconds

# Now we store ALL THREE channels. Each buffer holds the recent averages.
r_buffer = deque(maxlen=BUFFER_SIZE)
g_buffer = deque(maxlen=BUFFER_SIZE)
b_buffer = deque(maxlen=BUFFER_SIZE)


def pos_algorithm(r, g, b):
    """
    The POS method (Wang et al., 2017).
    Input: three equal-length arrays of R, G, B averages over time.
    Output: one 1D pulse waveform.
    """
    # Stack the three signals into a 3-row array: row0=R, row1=G, row2=B.
    rgb = np.array([r, g, b])                       # shape (3, N)

    # 1) Normalize each channel by its own mean (removes the DC/brightness level,
    #    keeps only the fluctuations).
    mean_rgb = np.mean(rgb, axis=1, keepdims=True)
    rgb_normalized = rgb / mean_rgb

    # 2) POS projection: two fixed combinations of the normalized channels.
    #    These specific weights are the "plane orthogonal to skin".
    s1 = rgb_normalized[1] - rgb_normalized[2]              # G - B
    s2 = rgb_normalized[1] + rgb_normalized[2] - 2 * rgb_normalized[0]  # G + B - 2R

    # 3) Combine s1 and s2, scaled so lighting/motion cancels out.
    #    alpha balances the two projections using their standard deviations.
    alpha = np.std(s1) / (np.std(s2) + 1e-9)               # +tiny number avoids /0
    pulse = s1 + alpha * s2

    # 4) Remove the mean of the final pulse so it's centered on zero.
    pulse = pulse - np.mean(pulse)
    return pulse


print("Collecting signal, then showing your pulse wave. Press 'q' to quit.")

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
        fx = x + int(w * 0.25)
        fy = y + int(h * 0.10)
        fw = int(w * 0.50)
        fh = int(h * 0.20)
        roi = frame[fy:fy + fh, fx:fx + fw]

        avg_color = np.mean(roi, axis=(0, 1))
        b_buffer.append(avg_color[0])
        g_buffer.append(avg_color[1])
        r_buffer.append(avg_color[2])

        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

    # A separate black canvas to draw the pulse waveform on.
    wave_img = np.zeros((200, BUFFER_SIZE, 3), dtype=np.uint8)

    # Only run POS once we have a full buffer.
    if len(g_buffer) == BUFFER_SIZE:
        pulse = pos_algorithm(
            np.array(r_buffer), np.array(g_buffer), np.array(b_buffer)
        )

        # Scale the pulse so it fits nicely in the 200-pixel-tall canvas.
        p_min, p_max = np.min(pulse), np.max(pulse)
        if p_max - p_min > 1e-9:
            pulse_scaled = (pulse - p_min) / (p_max - p_min)   # 0..1
            pulse_scaled = pulse_scaled * 180 + 10             # 10..190 px

            # Draw the waveform by connecting consecutive points.
            for i in range(1, len(pulse_scaled)):
                x1, y1 = i - 1, int(200 - pulse_scaled[i - 1])
                x2, y2 = i,     int(200 - pulse_scaled[i])
                cv2.line(wave_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.putText(wave_img, "Pulse wave (POS)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    else:
        # Still filling — show progress on the wave canvas.
        msg = f"Collecting... {len(g_buffer)} / {BUFFER_SIZE}"
        cv2.putText(wave_img, msg, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow("Camera - press q to quit", frame)
    cv2.imshow("Pulse Waveform", wave_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Closed.")