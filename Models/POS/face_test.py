import cv2

# OpenCV ships with a ready-made face detector (a "Haar cascade").
# This loads it. No training needed on our part.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open the webcam.")
    exit()

print("Looking for your face. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # The detector works on grayscale, so make a gray copy of the frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find faces. Returns a list of boxes: each is (x, y, width, height).
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    # Draw a green rectangle around each face we found.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Test - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Closed.")