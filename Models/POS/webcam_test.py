import cv2

# Open the default webcam. On Windows, CAP_DSHOW helps it start reliably.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Safety check: did the camera actually open?
if not cap.isOpened():
    print("ERROR: Could not open the webcam.")
    exit()

print("Webcam is on. Click the video window, then press 'q' to quit.")

while True:
    # Grab one frame (one image) from the camera.
    ret, frame = cap.read()

    # If we didn't get a frame, stop.
    if not ret:
        print("ERROR: Could not read a frame.")
        break

    # Show the frame in a window.
    cv2.imshow("Webcam Test - press q to quit", frame)

    # Wait 1 ms for a key press. If it's 'q', leave the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up: release the camera and close the window.
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")