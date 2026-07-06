import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open the webcam.")
    exit()

print("Measuring forehead color. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    if len(faces) > 0:
        # Use the first face found.
        (x, y, w, h) = faces[0]

        # Carve out a forehead patch from the face box.
        # Horizontally: middle 50% of the face width (skip the sides).
        # Vertically: the upper part, just below the top of the box.
        fx = x + int(w * 0.25)          # forehead left edge
        fy = y + int(h * 0.10)          # forehead top edge
        fw = int(w * 0.50)              # forehead width
        fh = int(h * 0.20)              # forehead height

        # Pull out just those pixels (this small image = our ROI).
        roi = frame[fy:fy + fh, fx:fx + fw]

        # Average the color over every pixel in the ROI.
        # OpenCV order is Blue, Green, Red.
        avg_color = np.mean(roi, axis=(0, 1))
        b, g, r = avg_color[0], avg_color[1], avg_color[2]

        # Print the three numbers, tidy to 2 decimals.
        print(f"B: {b:6.2f}   G: {g:6.2f}   R: {r:6.2f}")

        # Draw the forehead box in green and the full face box thin gray.
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)

    cv2.imshow("Color Test - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Closed.")