import cv2
import mediapipe as mp
import sys
import numpy as np

class HandTracking:
    FRAME_WIDTH = 800
    FRAME_HEIGHT = 300
    # Constants for GUI layout as percentages
    LEFT_COLUMN_X = 0.2  # Position of the finger names
    ACTUAL_ANGLES_X = 0.35  # Start position of the actual angle bars
    ACTUAL_PERCENT_X = 0.475  # Position of the actual angle percentages
    PREDICTED_ANGLES_X = 0.7  # Start position of the predicted angle bars
    PREDICTED_PERCENT_X = 0.825  # Position of the predicted angle percentages
    
    HEADER_Y_OFFSET = 0.1  # Offset for header text
    SECTION_START_Y_OFFSET = 0.25  # Vertical offset for the non-header section
    LINE_Y_OFFSET = 0.15  # Offset for drawing the bars

    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 1
    TEXT_THICKNESS = 2
    PERCENT_SCALE = 0.7
    PERCENT_THICKNESS = 2
    LINE_THICKNESS = 5
    RECT_THICKNESS = 2

    def __init__(self):
        # Initialize Mediapipe hand detection and drawing utilities
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = None

        # Define joint sets for each finger and their angle thresholds
        self.joint_sets = {
            'thumb':    [(4, 2, 0)],
            'index':    [(8, 6, 0)],
            'middle':   [(12, 10, 0)],
            'ring':     [(16, 14, 0)],
            'pinky':    [(20, 18, 0)],
        }

        self.joint_angle_thresholds = {
            'thumb': (132.5, 157.5),
            'index': (25, 165),
            'middle': (15, 167.5),
            'ring': (15, 167.5),
            'pinky': (15, 167.5)
        }

        # Initialize dictionaries to store last known angles and percentages for each finger
        self.last_known_angles = {finger: 0 for finger in self.joint_sets.keys()}
        self.last_known_percentages = {finger: 0 for finger in self.joint_sets.keys()}

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
    
    def get_percent(self, finger):
        """Get the last known percentage for the specified finger."""
        return self.last_known_percentages[finger]

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
        percentages_dict = {}
        
        for idx, (finger, joints) in enumerate(self.joint_sets.items()):
            angles = [self.calculate_angle(
                [hand_landmarks.landmark[j[0]].x, hand_landmarks.landmark[j[0]].y, hand_landmarks.landmark[j[0]].z],
                [hand_landmarks.landmark[j[1]].x, hand_landmarks.landmark[j[1]].y, hand_landmarks.landmark[j[1]].z],
                [hand_landmarks.landmark[j[2]].x, hand_landmarks.landmark[j[2]].y, hand_landmarks.landmark[j[2]].z],
            ) for j in joints]
            
            # Average angle of the joints in the finger
            avg_angle = np.mean(angles)
            
            # Calculate percentage
            percentage = int(self.angle_to_percentage(finger, avg_angle))
            
            # Update dictionaries
            angles_dict[finger] = avg_angle
            percentages_dict[finger] = percentage
        
        return angles_dict, percentages_dict

    def draw_angles(self, finger_frame, frame_width, frame_height, predicted_angles, terminal_out=True):
        """Draw the angles and percentages on the given frame."""
        if terminal_out:
            print('---------------------')
            
        for idx, (finger, avg_angle) in enumerate(self.last_known_angles.items()):
            percentage = self.last_known_percentages[finger]
            y_offset = int(frame_height * (self.SECTION_START_Y_OFFSET + idx * self.LINE_Y_OFFSET))

            # Draw finger name
            text_size = cv2.getTextSize(f"{finger.capitalize()}", self.TEXT_FONT, self.TEXT_SCALE, self.TEXT_THICKNESS)[0]
            text_x = int(frame_width * self.LEFT_COLUMN_X) - text_size[0]
            text_y = y_offset + (text_size[1] // 2)  # Center the text vertically
            cv2.putText(finger_frame, f"{finger.capitalize()}", 
                        (text_x, text_y), self.TEXT_FONT, self.TEXT_SCALE, (255, 255, 255), self.TEXT_THICKNESS)
            
            # Draw actual angle bars and rectangles
            self.draw_angle_bar(finger_frame, frame_width, y_offset, percentage, self.ACTUAL_ANGLES_X, (0, 255, 0))
            self.draw_percentage_text(finger_frame, frame_width, y_offset, percentage, self.ACTUAL_PERCENT_X, (0, 255, 0))

            # Print actual angles and optionally predicted angles
            if terminal_out:
                print(f"{finger.capitalize():<10} | {int(avg_angle):>3}° | {int(percentage):>3}%")
            
            if finger in predicted_angles:
                predicted_angle = predicted_angles[finger]
                predicted_percentage = int(self.angle_to_percentage(finger, predicted_angle))
                if terminal_out:
                    print(f"{'Prediction':<10} | {int(predicted_angle):>3}° | {predicted_percentage:>3}%")
                
                # Draw predicted angles
                self.draw_angle_bar(finger_frame, frame_width, y_offset, predicted_percentage, self.PREDICTED_ANGLES_X, (0, 0, 255))
                self.draw_percentage_text(finger_frame, frame_width, y_offset, predicted_percentage, self.PREDICTED_PERCENT_X, (0, 0, 255))
            if terminal_out:
                print('---------------------')

    def draw_angle_bar(self, frame, frame_width, y_offset, percentage, start_x, color):
        """Draw an angle bar and its surrounding rectangle on the frame."""
        line_start = (int(frame_width * start_x), y_offset)
        line_end = (line_start[0] + int(frame_width * 0.1 * percentage / 100), y_offset)
        cv2.line(frame, line_start, line_end, color, self.LINE_THICKNESS)
        cv2.rectangle(frame, 
                      (line_start[0] - 5, y_offset - 10), 
                      (line_start[0] + int(frame_width * 0.1) + 5, y_offset + 10), 
                      (255, 255, 255), self.RECT_THICKNESS)

    def draw_percentage_text(self, frame, frame_width, y_offset, percentage, start_x, color):
        """Draw the percentage text on the frame."""
        text = f"{percentage}%"
        text_size = cv2.getTextSize(text, self.TEXT_FONT, self.PERCENT_SCALE, self.PERCENT_THICKNESS)[0]
        full_text_x = cv2.getTextSize('100%', self.TEXT_FONT, self.PERCENT_SCALE, self.PERCENT_THICKNESS)[0][0]
        text_x = int(frame_width * start_x + full_text_x - text_size[0])
        text_y = y_offset + (text_size[1] // 2)  # Center the text vertically
        cv2.putText(frame, text, (text_x, text_y), self.TEXT_FONT, self.PERCENT_SCALE, color, self.PERCENT_THICKNESS)

    def update(self, draw_camera=True, draw_angles=True, terminal_out=False, predicted_angles={}):
        """Capture frames from the camera, process hand landmarks, and display the results."""
        if self.cap is None or not self.cap.isOpened():
            print("Camera is not opened.")
            return

        success, image = self.cap.read()
        if not success:
            print("Failed to capture image.")
            return

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = self.hands.process(image_rgb)

        # Clear the terminal output
        if terminal_out:
            sys.stdout.write("\033[H\033[J")

        # Create a blank image for finger angles with conditional width
        frame_width = self.FRAME_WIDTH
        frame_height = self.FRAME_HEIGHT
        
        if predicted_angles:
            finger_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        else:
            finger_frame = np.zeros((frame_height, 3*(frame_width)//5, 3), dtype=np.uint8)
            

        # Draw hand landmarks and connections
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Calculate angles for each finger and map to servo positions
                angles, percentages = self.get_calculated_angles(hand_landmarks)
                
                # Update last known values
                for finger in self.joint_sets.keys():
                    self.last_known_angles[finger] = angles[finger]
                    self.last_known_percentages[finger] = percentages[finger]
                    
        # Draw the last known values
        self.draw_angles(finger_frame, frame_width, frame_height, predicted_angles, terminal_out=terminal_out)

        # Add labels above columns
        actual_text = "Target"
        predicted_text = "Predicted"
        
        actual_text_size = cv2.getTextSize(actual_text, self.TEXT_FONT, self.TEXT_SCALE, self.TEXT_THICKNESS)[0]
        predicted_text_size = cv2.getTextSize(predicted_text, self.TEXT_FONT, self.TEXT_SCALE, self.TEXT_THICKNESS)[0]
        
        actual_text_x = int(frame_width * self.ACTUAL_ANGLES_X + (frame_width * 0.1 / 2) - (actual_text_size[0] / 2))
        if predicted_angles:
            predicted_text_x = int(frame_width * self.PREDICTED_ANGLES_X + (frame_width * 0.1 / 2) - (predicted_text_size[0] / 2))
        
        # Draw the header text
        cv2.putText(finger_frame, actual_text, (int(actual_text_x), int(frame_height * self.HEADER_Y_OFFSET)), self.TEXT_FONT, self.TEXT_SCALE, (0, 255, 0), self.TEXT_THICKNESS)
        
        # Draw predicted text if predicted angles are provided
        if predicted_angles:
            cv2.putText(finger_frame, predicted_text, (int(predicted_text_x), int(frame_height * self.HEADER_Y_OFFSET)), self.TEXT_FONT, self.TEXT_SCALE, (0, 0, 255), self.TEXT_THICKNESS)
        
        # Display the images
        if draw_camera:
            cv2.imshow('MediaPipe Hands', image)
        if draw_angles:
            cv2.imshow('Finger Angles', finger_frame)
        
        # Close the windows if the 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            self.close()

def main():
    hand_tracker = HandTracking()
    hand_tracker.start()

    try:
        while hand_tracker.isOpened():
            hand_tracker.update(draw_camera=True, draw_angles=True, terminal_out=True, predicted_angles={'thumb': 150, 'index': 30, 'middle': 20, 'ring': 20, 'pinky': 20})
    finally:
        hand_tracker.close()

if __name__ == "__main__":
    main()
