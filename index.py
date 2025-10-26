import cv2
import time
from datetime import datetime, timedelta

class FaceDetectionCrossing:
    def __init__(self, min_faces=1, crossing_duration=20):
        """
        Initialize face detection crossing system.
        
        Args:
            min_faces (int): Minimum number of faces required to trigger crossing
            crossing_duration (int): Duration in seconds to keep isCrossing True
        """
        self.min_faces = min_faces
        self.crossing_duration = crossing_duration
        self.isCrossing = False
        self.crossing_end_time = None
        
        # Load the Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect_faces(self, frame):
        """
        Detect faces in a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            list: List of detected face rectangles
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def update_crossing_status(self, num_faces):
        """
        Update the isCrossing status based on number of detected faces.
        
        Args:
            num_faces (int): Number of faces detected in current frame
        """
        current_time = datetime.now()
        
        # If minimum faces detected, set crossing to True and start timer
        if num_faces >= self.min_faces:
            if not self.isCrossing:
                self.isCrossing = True
                print(f"Crossing activated: {num_faces} faces detected (min: {self.min_faces})")
            
            # Reset the timer - keep extending while faces are present
            self.crossing_end_time = current_time + timedelta(seconds=self.crossing_duration)
        
        # If crossing is active and timer has expired, deactivate
        elif self.isCrossing and self.crossing_end_time:
            if current_time >= self.crossing_end_time:
                self.isCrossing = False
                self.crossing_end_time = None
                print("Crossing deactivated: timer expired")
    
    def process_frame(self, frame):
        """
        Process a frame: detect faces and update crossing status.
        
        Args:
            frame: Input image frame
            
        Returns:
            tuple: (processed_frame, number_of_faces, isCrossing)
        """
        # Detect faces
        faces = self.detect_faces(frame)
        num_faces = len(faces)
        
        # Update crossing status
        self.update_crossing_status(num_faces)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add status text to frame
        status_text = f"Faces: {num_faces} | Crossing: {self.isCrossing}"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.isCrossing and self.crossing_end_time:
            time_remaining = (self.crossing_end_time - datetime.now()).total_seconds()
            if time_remaining > 0:
                timer_text = f"Time remaining: {time_remaining:.1f}s"
                cv2.putText(frame, timer_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, num_faces, self.isCrossing

def main():
    """
    Main function to run the face detection crossing system.
    """
    # Initialize the face detection crossing system
    # min_faces: Minimum number of faces to activate crossing
    # crossing_duration: Duration in seconds to keep crossing active
    detector = FaceDetectionCrossing(min_faces=1, crossing_duration=20)
    
    # Open webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Face Detection Crossing System Started")
    print(f"Minimum faces required: {detector.min_faces}")
    print(f"Crossing duration: {detector.crossing_duration} seconds")
    print("Press 'q' to quit")
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process the frame
            processed_frame, num_faces, is_crossing = detector.process_frame(frame)
            
            # Display the resulting frame
            cv2.imshow('Face Detection Crossing', processed_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Face Detection Crossing System Stopped")

if __name__ == "__main__":
    main()
