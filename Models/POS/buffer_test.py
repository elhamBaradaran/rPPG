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

# --- Our sliding buffer ---
# We assume ~30 frames per second. To hold 10 seconds, we need 300 frames.
FPS = 30
WINDOW_SECONDS = 10
BUFFER_SIZE = FPS * WINDOW_SECONDS   # = 300

# A deque with maxlen automatically drops the oldest item when full.
# We store the Green average of each frame (green carries the pulse best).
green_buffer = deque(maxlen=BUFFER_SIZE)

print("Filling the buffer. Hold still and look at the camera. Press 'q' to quit.")

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
        green_value = avg_color[1]        # index 1 = Green

        # Drop the newest Green value into the buffer.
        green_buffer.append(green_value)

        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

    # Show how full the buffer is, as text on the video.
    status = f"Buffer: {len(green_buffer)} / {BUFFER_SIZE}"
    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Buffer Test - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Closed. Buffer ended with {len(green_buffer)} values.")