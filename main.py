import cv2
import numpy as np
from datetime import datetime

class FaceDetectionCrossing:
    def __init__(self, min_faces=1):
        self.min_faces = min_faces
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces


def overlay_image(base, overlay, pos=(0, 0)):
    """Overlay transparent image onto another."""
    x, y = pos
    h, w = overlay.shape[:2]

    if y + h > base.shape[0] or x + w > base.shape[1]:
        return base  # Skip if out of bounds

    if overlay.shape[2] == 4:  # has alpha channel
        alpha_overlay = overlay[:, :, 3] / 255.0
        alpha_base = 1.0 - alpha_overlay
        for c in range(0, 3):
            base[y:y+h, x:x+w, c] = (
                alpha_overlay * overlay[:, :, c] + alpha_base * base[y:y+h, x:x+w, c]
            )
    else:
        base[y:y+h, x:x+w] = overlay

    return base


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetectionCrossing(min_faces=1)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    # Load assets
    base_img = cv2.imread("assets/stoplight.png", cv2.IMREAD_UNCHANGED)
    light_red = cv2.imread("assets/red.png", cv2.IMREAD_UNCHANGED)
    light_yellow = cv2.imread("assets/yellow.png", cv2.IMREAD_UNCHANGED)
    light_green = cv2.imread("assets/green.png", cv2.IMREAD_UNCHANGED)


    # Light timing setup
    RED_DURATION = 30
    GREEN_DURATION = 30
    YELLOW_DURATION = 1
    NO_FACE_LIMIT = 10  # shorten red after 10s if no face

    state = "red"
    previous_state = "red"
    last_change = datetime.now()
    last_face_time = datetime.now()

    print("Traffic Light Started (Press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        num_faces = len(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        now = datetime.now()
        elapsed = (now - last_change).total_seconds()

        if num_faces > 0:
            last_face_time = now

        # ---- LOGIC ----
        if state == "red":
            no_face_time = (now - last_face_time).total_seconds()
            if (no_face_time >= NO_FACE_LIMIT and elapsed >= NO_FACE_LIMIT) or elapsed >= RED_DURATION:
                previous_state = "red"
                state = "yellow"
                last_change = now

        elif state == "yellow":
            if elapsed >= YELLOW_DURATION:
                if previous_state == "red":
                    state = "green"
                else:
                    state = "red"
                previous_state = state
                last_change = now

        elif state == "green":
            if elapsed >= GREEN_DURATION:
                previous_state = "green"
                state = "yellow"
                last_change = now

        # ---- TIMER ----
        if state == "red":
            remaining = RED_DURATION - elapsed if elapsed < RED_DURATION else 0
        elif state == "green":
            remaining = GREEN_DURATION - elapsed if elapsed < GREEN_DURATION else 0
        else:
            remaining = YELLOW_DURATION - elapsed if elapsed < YELLOW_DURATION else 0

        timer_text = f"{state.upper()} - {int(max(remaining, 0))}s"
        cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # ---- STOPLIGHT OVERLAY ----
        h, w, _ = frame.shape
        light_x = w - 100
        light_y = 20

        stoplight = base_img.copy()
        if state == "red":
            stoplight = overlay_image(stoplight, light_red)
        elif state == "yellow":
            stoplight = overlay_image(stoplight, light_yellow)
        elif state == "green":
            stoplight = overlay_image(stoplight, light_green)

        frame = overlay_image(frame, stoplight, (light_x, light_y))

        # ---- FACE INFO ----
        info = f"Faces: {num_faces}"
        cv2.putText(frame, info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        # ---- SHOW FRAME ----
        cv2.imshow("Face Detection Crossing", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped")


if __name__ == "__main__":
    main()
