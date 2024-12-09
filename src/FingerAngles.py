import cv2
import mediapipe as mp
import numpy as np
import sys
import time

RESOLUTION = (1920, 1080)
FRAME_RATE = 120

class FingerAngles:

    def __init__(self, max_num_hands=1, min_detection_confidence=0.5):
        # Initialize Mediapipe hand detection and drawing utilities
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence, model_complexity=0)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = None
        
        self.fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

        # Define joint sets for each finger and their angle thresholds
        self.joint_sets = {
            'thumb':    [((4, 3, 0))],
            'index':    [(8, 5, 0)],
            'middle':   [(12, 9, 0)],
            'ring':     [(16, 13, 0)],
            'pinky':    [(20, 17, 0)],
        }

        self.joint_angle_thresholds = {
                    'thumb':    (120, 150),
                    'index':    (90, 120),
                    'middle':   (90, 145),
                    'ring':     (70, 145),
                    'pinky':    (80, 157.5)
        }

        # Initialize dictionaries to store last known angles and percentages for each finger
        self.last_known_angles = {finger: 0 for finger in self.joint_sets.keys()}
        self.found_hand = False

    def start(self):
        """Start the video capture."""
        self.cap = cv2.VideoCapture(0)

    def close(self):
        """Release the video capture and close all OpenCV windows."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
    def isOpened(self):
        """Check if the video capture is open."""
        return self.cap.isOpened()
    
    def __getitem__(self, finger):
        return {'angle': self.get_angle(finger), 'percentage': self.get_percent(finger)}
    
    def get_angle(self, finger):
        """Get the last known angle for the specified finger."""
        return self.last_known_angles[finger]
        
    def get_angle_list(self):
        """Get the last known percentages for all fingers."""
        return [self.get_angle(finger) for finger in self.fingers]
        
    def get_angle_dict(self):
        """Get the last known angles for all fingers."""
        return {finger: self.get_angle(finger) for finger in self.fingers}
        
    def get_percent(self, finger):
        """Get the last known percentage for the specified finger."""
        return self.angle_to_percentage(finger, self.get_angle(finger))
        
    def get_percent_list(self):
        """Get the last known percentages for all fingers."""
        return [self.get_percent(finger) for finger in self.fingers]
    
    def get_percent_dict(self):
        """Get the last known percentages for all fingers."""
        return {finger: self.get_percent(finger) for finger in self.fingers}

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points a, b, and c."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arccos(np.dot(a-b, c-b) / (np.linalg.norm(a-b) * np.linalg.norm(c-b)))
        return np.degrees(radians)

    def angle_to_percentage(self, finger, angle):
        """Convert a given angle to the corresponding percentage for a specified finger."""
        joint_thresholds = self.joint_angle_thresholds[finger]
        clipped_angle = np.clip(angle, joint_thresholds[0], joint_thresholds[1])
        normalized_angle = (clipped_angle - joint_thresholds[0]) / (joint_thresholds[1] - joint_thresholds[0])
        return normalized_angle * 100

    def percentage_to_angle(self, finger, percentage):
        """Convert a given percentage back to the corresponding angle for a specified finger."""
        joint_thresholds = self.joint_angle_thresholds[finger]
        return (percentage / 100) * (joint_thresholds[1] - joint_thresholds[0]) + joint_thresholds[0]

    def get_calculated_angles(self, hand_landmarks):
        """Calculate the angles and corresponding percentages for each finger based on hand landmarks."""
        angles_dict = {}
        
        for finger, joints in self.joint_sets.items():
            angles = [self.calculate_angle(
                [hand_landmarks.landmark[j[0]].x, hand_landmarks.landmark[j[0]].y, hand_landmarks.landmark[j[0]].z],
                [hand_landmarks.landmark[j[1]].x, hand_landmarks.landmark[j[1]].y, hand_landmarks.landmark[j[1]].z],
                [hand_landmarks.landmark[j[2]].x, hand_landmarks.landmark[j[2]].y, hand_landmarks.landmark[j[2]].z],
            ) for j in joints] # Calculate angles for each joint set
            
            
            # Update dictionary with average angle for each finger
            angles_dict[finger] = np.mean(angles)
        
        return angles_dict

    def print_angles(self, terminal_out=True):
        """Draw the angles and percentages on the given frame."""
        if terminal_out:
            print('---------------------')
            
            for finger in self.fingers:
                angle = self.get_angle(finger)
                percentage = self.get_percent(finger)
                print(f"{finger.capitalize():<10} | {int(angle):>3}° | {int(percentage):>3}%")

                print('---------------------')

    def update(self, draw_camera=True, draw_landmarks=True, terminal_out=False, clear_terminal=True):
        """Capture frames from the camera, process hand landmarks, and display the results."""
        if self.cap is None or not self.cap.isOpened():
            print("Camera is not opened.")
            return

        success, image = self.cap.read()
        if not success:
            print("Failed to capture image.")
            return
        
        # Print the framerate
        # print(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        # Crop the image to the center square
        height, width, _ = image.shape
        side = min(height, width)
        
        # image = image[side//2:height-side//2, side//2:width-side//2]
        image = image[0:height, 0:width]
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_rgb = cv2.resize(image_rgb, (224, 224))

        # Process the image and detect hands
        results = self.hands.process(image_rgb)
            
        # Draw hand landmarks and connections
        if results.multi_hand_landmarks:
            self.found_hand = True
            for hand_landmarks in results.multi_hand_landmarks:
                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Calculate angles for each finger and map to servo positions
                angles = self.get_calculated_angles(hand_landmarks)
                
                # Update last known values
                for finger in angles.keys():
                    self.last_known_angles[finger] = angles[finger]
                    
            # Clear the terminal output
            if terminal_out and clear_terminal:
                sys.stdout.write("\033[H\033[J")
                
            # Print the angles and percentages
            if terminal_out:
                self.print_angles(terminal_out=True)
        else :
            self.found_hand = False
            if terminal_out:
                print("No hand detected.")
        
        
        # Display the images
        if draw_camera:
            cv2.imshow('MediaPipe Hands', image)
        
        # Close the windows if the 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            self.close()
            
        return image

def main():
    fingers = FingerAngles()
    fingers.start()
    last_fps_time = time.time()
    frame_count = 0
    try:
        while fingers.isOpened():
            fingers.update(draw_camera=True, terminal_out=True)
            frame_count += 1
            if time.time() - last_fps_time >= 1:
                print(f"FPS: {frame_count}")
                frame_count = 0
                last_fps_time = time.time()
    finally:
        fingers.close()

if __name__ == "__main__":
    main()
