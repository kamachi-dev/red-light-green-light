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

class CarDetection:
    def __init__(self):
        self.car_cascade = cv2.CascadeClassifier("cars_haar.xml")  # <<< MODIFY: your car cascade path

    def detect_cars(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
        )
        return cars

def overlay_image(base, overlay, pos=(0, 0)):
    x, y = pos
    h, w = overlay.shape[:2]

    if y + h > base.shape[0] or x + w > base.shape[1]:
        return base

    if overlay.shape[2] == 4:  
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
    cap = cv2.VideoCapture("newest.mp4")
    # cap = cv2.VideoCapture("D:\\Recordings (2023)\\New Recordings\\2025-11-17 21-07-10.mp4")  
    # cap = cv2.VideoCapture("D:\\Recordings (2023)\\New Recordings\\2025-11-17 20-21-24.mp4")  
    detector = FaceDetectionCrossing(min_faces=1)
    car_detector = CarDetection()

    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    # Load assets
    base_img = cv2.imread("assets/stoplight.png", cv2.IMREAD_UNCHANGED)
    light_red = cv2.imread("assets/red.png", cv2.IMREAD_UNCHANGED)
    light_yellow = cv2.imread("assets/yellow.png", cv2.IMREAD_UNCHANGED)
    light_green = cv2.imread("assets/green.png", cv2.IMREAD_UNCHANGED)

    # Light timing
    RED_DURATION = 25
    GREEN_DURATION = 25
    YELLOW_DURATION = 2
    NO_FACE_LIMIT = 5
    NO_CAR_LIMIT = 5  

    green_started = False
    red_started = False
    state = "red"
    previous_state = "red"
    last_change = datetime.now()
    last_face_time = datetime.now()
    last_car_time = datetime.now()
    green_no_car_start = datetime.now() 

    print("Traffic Light Started (Press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (int(1920 * 0.6), int(1080 * 0.6)))
        now = datetime.now()
        elapsed = (now - last_change).total_seconds()

        # ---- FACE DETECTION ----
        faces = detector.detect_faces(frame)
        num_faces = len(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if num_faces > 0:
            last_face_time = now

        # ---- CAR DETECTION ----
        cars = car_detector.detect_cars(frame)
        cars = [(x, y, w, h) for (x, y, w, h) in cars if y > 450]

        # <<< MODIFIED: remove cars overlapping faces >>>
        filtered_cars = []
        for (cx, cy, cw, ch) in cars:
            overlap = False
            for (fx, fy, fw, fh) in faces:
                if (cx < fx + fw and cx + cw > fx and cy < fy + fh and cy + ch > fy):
                    overlap = True
                    break
            if not overlap:
                filtered_cars.append((cx, cy, cw, ch))
        cars = filtered_cars

        num_cars = len(cars)
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if num_cars > 0:
            last_car_time = now

        # ---- LOGIC ----
        if state == "red":
            if not red_started:
                red_no_face_start = now  
                red_started = True

            if num_faces > 0:
                red_no_face_start = now

            no_face_elapsed = (now - red_no_face_start).total_seconds()
            print(f"current state: {state}: {previous_state}: {no_face_elapsed}")
            print(no_face_elapsed >= NO_FACE_LIMIT and num_cars > 0)

            if no_face_elapsed >= NO_FACE_LIMIT and num_cars > 0:
                previous_state = "red"
                state = "yellow"
                last_change = now
                red_started = False 
       
            elif elapsed >= RED_DURATION:
                previous_state = "red"
                state = "yellow"
                last_change = now
                red_started = False

        elif state == "yellow":
            print(f"current state: {state}: {previous_state}")
            if elapsed >= YELLOW_DURATION:
                if previous_state == "red":
                    state = "green"
                else:
                    state = "red"
                previous_state = "yellow"

                last_change = now

        elif state == "green":

            if not green_started:
                green_no_car_start = now
                green_started = True

            if num_cars > 0:
                green_no_car_start = now

            no_car_elapsed = (now - green_no_car_start).total_seconds()

            if no_car_elapsed >= NO_CAR_LIMIT and num_faces > 0:
                previous_state = "green"
                state = "yellow"
                last_change = now
                green_started = False 

            elif elapsed >= GREEN_DURATION:
                previous_state = "green"
                state = "yellow"
                last_change = now
                green_started = False  

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

        # ---- INFO ----
        info = f"Faces: {num_faces} | Cars: {num_cars}"
        cv2.putText(frame, info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.imshow("Face + Car Detection Crossing", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped")

if __name__ == "__main__":
    main()
