import cv2
import mediapipe as mp
import sys
import numpy as np
import pandas as pd
import time
import pylsl
from pylsl import StreamInlet, resolve_stream, proc_ALL
import threading

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
        self.last_known_angles = {finger: np.nan for finger in self.joint_sets.keys()}
        self.last_known_percentages = {finger: np.nan for finger in self.joint_sets.keys()}

    def start(self):
        """Start the video capture."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

    def close(self):
        """Release the video capture and close all OpenCV windows."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def isOpened(self):
        """Check if the video capture is open."""
        return self.cap.isOpened()

    def get_angle(self, finger):
        """Get the last known angle for the specified finger."""
        return self.last_known_angles[finger]

    def get_percent(self, finger):
        """Get the last known percentage for the specified finger."""
        return self.last_known_percentages[finger]

    def get_current_data(self):
        """Get the current angles and percentages for all fingers."""
        return self.last_known_angles.copy(), self.last_known_percentages.copy()

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points a, b, and c."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arccos(np.clip(np.dot(a-b, c-b) / (np.linalg.norm(a-b) * np.linalg.norm(c-b)), -1.0, 1.0))
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
            angles = []
            for j in joints:
                try:
                    angle = self.calculate_angle(
                        [hand_landmarks.landmark[j[0]].x, hand_landmarks.landmark[j[0]].y, hand_landmarks.landmark[j[0]].z],
                        [hand_landmarks.landmark[j[1]].x, hand_landmarks.landmark[j[1]].y, hand_landmarks.landmark[j[1]].z],
                        [hand_landmarks.landmark[j[2]].x, hand_landmarks.landmark[j[2]].y, hand_landmarks.landmark[j[2]].z],
                    )
                    angles.append(angle)
                except Exception as e:
                    angles.append(np.nan)
            
            # Average angle of the joints in the finger
            avg_angle = np.nanmean(angles)
            
            # Calculate percentage
            if not np.isnan(avg_angle):
                percentage = self.angle_to_percentage(finger, avg_angle)
            else:
                percentage = np.nan
            
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
            
            if not np.isnan(percentage):
                # Draw actual angle bars and rectangles
                self.draw_angle_bar(finger_frame, frame_width, y_offset, percentage, self.ACTUAL_ANGLES_X, (0, 255, 0))
                self.draw_percentage_text(finger_frame, frame_width, y_offset, percentage, self.ACTUAL_PERCENT_X, (0, 255, 0))
            else:
                # Indicate that data is not available
                cv2.putText(finger_frame, "N/A", 
                            (int(frame_width * self.ACTUAL_PERCENT_X), y_offset), self.TEXT_FONT, self.PERCENT_SCALE, (0, 0, 255), self.PERCENT_THICKNESS)
            
            # Print actual angles and optionally predicted angles
            if terminal_out:
                angle_text = f"{int(avg_angle)}°" if not np.isnan(avg_angle) else "N/A"
                percent_text = f"{int(percentage)}%" if not np.isnan(percentage) else "N/A"
                print(f"{finger.capitalize():<10} | {angle_text:>5} | {percent_text:>5}")
            
            if finger in predicted_angles:
                predicted_angle = predicted_angles[finger]
                predicted_percentage = self.angle_to_percentage(finger, predicted_angle)
                if terminal_out:
                    print(f"{'Prediction':<10} | {int(predicted_angle):>3}° | {int(predicted_percentage):>3}%")
                
                # Draw predicted angles
                self.draw_angle_bar(finger_frame, frame_width, y_offset, predicted_percentage, self.PREDICTED_ANGLES_X, (0, 0, 255))
                self.draw_percentage_text(finger_frame, frame_width, y_offset, predicted_percentage, self.PREDICTED_PERCENT_X, (0, 0, 255))
            if terminal_out:
                print('---------------------')

    def draw_angle_bar(self, frame, frame_width, y_offset, percentage, start_x, color):
        """Draw an angle bar and its surrounding rectangle on the frame."""
        line_start = (int(frame_width * start_x), y_offset)
        bar_length = int(frame_width * 0.1 * percentage / 100) if not np.isnan(percentage) else 0
        line_end = (line_start[0] + bar_length, y_offset)
        cv2.line(frame, line_start, line_end, color, self.LINE_THICKNESS)
        cv2.rectangle(frame, 
                      (line_start[0] - 5, y_offset - 10), 
                      (line_start[0] + int(frame_width * 0.1) + 5, y_offset + 10), 
                      (255, 255, 255), self.RECT_THICKNESS)

    def draw_percentage_text(self, frame, frame_width, y_offset, percentage, start_x, color):
        """Draw the percentage text on the frame."""
        text = f"{int(percentage)}%" if not np.isnan(percentage) else "N/A"
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
                if draw_camera:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Calculate angles for each finger and map to servo positions
                angles, percentages = self.get_calculated_angles(hand_landmarks)
                
                # Update last known values
                for finger in self.joint_sets.keys():
                    self.last_known_angles[finger] = angles[finger]
                    self.last_known_percentages[finger] = percentages[finger]
        else:
            # No hands detected, set angles and percentages to NaN
            for finger in self.joint_sets.keys():
                self.last_known_angles[finger] = np.nan
                self.last_known_percentages[finger] = np.nan

        # Draw the last known values
        if draw_angles:
            self.draw_angles(finger_frame, frame_width, frame_height, predicted_angles, terminal_out=terminal_out)

        # Display the images
        if draw_camera:
            cv2.imshow('MediaPipe Hands', image)
        if draw_angles:
            cv2.imshow('Finger Angles', finger_frame)

        # Close the windows if the 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            self.close()

def collect_emg_data(inlet, emg_data_list, emg_channel_count, stop_event):
    """Collect EMG data at high frequency and convert LSL timestamps to Unix timestamps."""
    sample_count = 0
    lsl_to_unix_offset = time.time() - pylsl.local_clock()  # Initial LSL-to-Unix offset

    while not stop_event.is_set():
        try:
            # Pull a sample from the inlet
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample is None:
                continue

            # Convert LSL timestamp to Unix timestamp using the offset
            unix_timestamp = timestamp + lsl_to_unix_offset

            # Create data row with Unix timestamp and EMG data
            data_row = {'timestamp': unix_timestamp}
            for i in range(emg_channel_count):
                data_row[f'emg_channel_{i + 1}'] = sample[i]
            emg_data_list.append(data_row)

            # Periodically recalibrate the offset to account for drift
            sample_count += 1
            if sample_count % 1000 == 0:
                lsl_to_unix_offset = time.time() - pylsl.local_clock()
                
        except Exception as e:
            print(f"Error in EMG data collection: {e}")
            continue


def main():
    # Initialize hand tracker
    hand_tracker = HandTracking()
    try:
        hand_tracker.start()
    except IOError as e:
        print(e)
        return
    
    # Initialize EMG stream
    print("Looking for all available EMG streams...")
    streams = resolve_stream('type', 'EMG')
    
    if len(streams) == 0:
        print("No EMG streams found.")
        inlet = None
        emg_channel_count = 0
    else:
        print(f"{len(streams)} EMG stream(s) found:")
        for idx, stream in enumerate(streams):
            print(f"Stream {idx + 1}:")
            print(f"  Name: {stream.name()}")
            print(f"  Type: {stream.type()}")
            print(f"  Channel Count: {stream.channel_count()}")
            print(f"  Source ID: {stream.source_id()}")
            print(f"  Nominal Sampling Rate: {stream.nominal_srate()} Hz")
            print(f"  Hostname: {stream.hostname()}")

        # Select the first available stream
        inlet = StreamInlet(streams[0], processing_flags=pylsl.proc_clocksync)
        # inlet = StreamInlet(streams[0], processing_flags=proc_ALL)
        print("\nUsing the first EMG stream...")

        # Get EMG channel count from the StreamInfo object
        emg_channel_count = inlet.info().channel_count()

    # Create a pandas DataFrame to store data
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    data_columns = []
    for finger in fingers:
        data_columns.append(finger + '_angle')
        data_columns.append(finger + '_percent')
    data_columns.append('timestamp_hand')

    # Initialize data lists
    emg_data_list = []
    hand_data_list = []

    # Create a stop event for threads
    stop_event = threading.Event()

    # Start EMG data collection in a separate thread
    if inlet:
        emg_thread = threading.Thread(target=collect_emg_data, args=(inlet, emg_data_list, emg_channel_count, stop_event))
        emg_thread.start()

    # Initialize variables for refresh rate calculation
    loop_count = 0
    start_time = time.time()
    last_print_time = start_time

    try:
        while hand_tracker.isOpened():
            current_time = time.time()
            loop_count += 1

            # Initialize data row
            data_row = {}
            
            # Update hand tracker
            try:
                timestamp_hand = time.time()
                hand_tracker.update(draw_camera=False, draw_angles=True, terminal_out=False)
                angles, percentages = hand_tracker.get_current_data()
            except Exception as e:
                print(f"Error in hand tracking: {e}")
                angles, percentages = {}, {}
                timestamp_hand = np.nan

            # Collect hand tracking data
            for finger in fingers:
                angle = angles.get(finger, np.nan)
                percent = percentages.get(finger, np.nan)
                data_row[finger + '_angle'] = angle
                data_row[finger + '_percent'] = percent
            data_row['timestamp_hand'] = timestamp_hand

            # Append to hand data list
            hand_data_list.append(data_row)
            
            # Every second, print the refresh rate and data table size
            if current_time - last_print_time >= 1.0:
                refresh_rate = loop_count / (current_time - last_print_time)
                hand_data_size = len(hand_data_list)
                emg_data_size = len(emg_data_list)
                print(f"Hand Tracking Refresh Rate: {refresh_rate:.2f} Hz, Hand Data Size: {hand_data_size}, EMG Data Size: {emg_data_size}")
                # Reset counters
                loop_count = 0
                last_print_time = current_time
                
            # Break the loop if needed
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        hand_tracker.close()
        cv2.destroyAllWindows()
        stop_event.set()
        if inlet:
            emg_thread.join()
        # Save data to CSV
        hand_df = pd.DataFrame(hand_data_list)
        emg_df = pd.DataFrame(emg_data_list)
        hand_df.to_csv('hand_data.csv', index=False)
        emg_df.to_csv('emg_data.csv', index=False)
        print("Data saved to 'hand_data.csv' and 'emg_data.csv'")

if __name__ == "__main__":
    main()
