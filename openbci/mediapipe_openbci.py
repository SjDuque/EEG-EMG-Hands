import cv2
import mediapipe as mp
import numpy as np
import logging
import time
import sys
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from brainflow.exit_codes import BrainFlowError

# Constants for easy configuration
SERIAL_PORT = '/dev/cu.usbserial-D30DLPKS'
BOARD_ID = BoardIds.CYTON_BOARD.value
CHANNELS = [0, 1, 2, 3, 4]
DURATION_MS = 50  # Duration of information in milliseconds
SAMPLING_RATE = 250  # Cyton board's default sampling rate
SEQUENCE_LENGTH = (DURATION_MS * SAMPLING_RATE) // 1000  # Number of frames for 50 ms

# MediaPipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points in 3D space
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arccos(np.dot(a-b, c-b) / (np.linalg.norm(a-b) * np.linalg.norm(c-b)))
    return np.degrees(radians)

# Map the landmark indices to joint sets for each finger
joint_sets = {
    'thumb':    [(4, 2, 0)],
    'index':    [(8, 6, 0)],
    'middle':   [(12, 10, 0)],
    'ring':     [(16, 14, 0)],
    'pinky':    [(20, 18, 0)],
}

joint_angle_thresholds = {
    'thumb': (132.5, 157.5),
    'index': (25, 165),
    'middle': (15, 167.5),
    'ring': (15, 167.5),
    'pinky': (15, 167.5)
}

def main():
    logging.basicConfig(level=logging.INFO)

    # Setup BrainFlow parameters
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board_shim = BoardShim(BOARD_ID, params)
    
    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000)

        # Initialize the video capture
        cap = cv2.VideoCapture(0)

        # Initialize last known angles and percentages
        last_known_angles = {finger: 0 for finger in joint_sets.keys()}
        last_known_percentages = {finger: 0 for finger in joint_sets.keys()}

        # Graph setup
        exg_channels = BoardShim.get_exg_channels(BOARD_ID)
        sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
        update_speed_ms = 50
        window_size = 4
        num_points = window_size * sampling_rate
        window_name = "BrainFlow Plot"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_height = 600
        window_width = 800
        plot_data = np.zeros((len(exg_channels), num_points))

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image and detect hands
            results = hands.process(image_rgb)

            # Create a blank image for finger angles
            finger_frame = np.zeros((300, 800, 3), dtype=np.uint8)

            # Draw hand landmarks and connections
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Calculate angles for each finger and map to servo positions
                    for idx, (finger, joints) in enumerate(joint_sets.items()):
                        angles = [calculate_angle(
                            [hand_landmarks.landmark[j[0]].x, hand_landmarks.landmark[j[0]].y, hand_landmarks.landmark[j[0]].z],
                            [hand_landmarks.landmark[j[1]].x, hand_landmarks.landmark[j[1]].y, hand_landmarks.landmark[j[1]].z],
                            [hand_landmarks.landmark[j[2]].x, hand_landmarks.landmark[j[2]].y, hand_landmarks.landmark[j[2]].z],
                        ) for j in joints]

                        # Average angle of the joints in the finger
                        avg_angle = np.mean(angles)
                        joint_thresholds = joint_angle_thresholds[finger]

                        # Clip avg_angle to be within the threshold range
                        clipped_angle = np.clip(avg_angle, joint_thresholds[0], joint_thresholds[1])

                        # Normalize the angle to the threshold range
                        normalized_angle = (clipped_angle - joint_thresholds[0]) / (joint_thresholds[1] - joint_thresholds[0])

                        # Calculate the percentage
                        percentage = int(normalized_angle * 100)

                        # Update last known values
                        last_known_angles[finger] = avg_angle
                        last_known_percentages[finger] = percentage

            # Draw the last known values
            for idx, (finger, avg_angle) in enumerate(last_known_angles.items()):
                percentage = last_known_percentages[finger]
                cv2.putText(finger_frame, f"{finger.capitalize()}", (50, 50 + idx * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                line_start = (200, 45 + idx * 50)
                line_end = (200 + percentage, 45 + idx * 50)
                cv2.line(finger_frame, line_start, line_end, (0, 255, 0), 5)
                # White perimeter around the full range where the green line can fill
                cv2.rectangle(finger_frame,
                              (200, 38 + idx * 50),
                              (380, 52 + idx * 50),
                              (255, 255, 255), 2)
                # Display the percentage next to the bar, right aligned
                text = f"{percentage}%"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = 460 - text_size[0]  # Move the text further to the right and right-align it
                cv2.putText(finger_frame, text, (text_x, 50 + idx * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                print('---------------------------------')
                print(f"{finger.capitalize()}\tRaw Angle: {avg_angle:.2f}\tPercentage: {percentage}%")
                print('---------------------------------')

            # Update BrainFlow data
            data = board_shim.get_current_board_data(num_points)
            if data is not None and data.shape[1] > 0:
                actual_points = data.shape[1]
                for count, channel in enumerate(exg_channels):
                    channel_data = data[channel][:actual_points]
                    if len(channel_data) == 0:
                        logging.warning(f'No data for channel {channel}')
                        continue

                    try:
                        # Process data
                        DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
                        DataFilter.perform_bandpass(channel_data, sampling_rate, 3.0, 45.0, 2,
                                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                        DataFilter.perform_bandstop(channel_data, sampling_rate, 48.0, 52.0, 2,
                                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                        DataFilter.perform_bandstop(channel_data, sampling_rate, 58.0, 62.0, 2,
                                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                        plot_data[count, :actual_points] = channel_data
                    except BrainFlowError as e:
                        logging.error(f'Error processing data for channel {channel}: {e}')
                        continue

                # Plot BrainFlow data
                img = np.zeros((window_height, window_width, 3), dtype=np.uint8)
                for i in range(len(exg_channels)):
                    y = plot_data[i, :actual_points] - np.min(plot_data[i, :actual_points])
                    if np.max(y) != 0:
                        y = y / np.max(y) * (window_height // len(exg_channels))
                    y = y.astype(np.int32)
                    x = np.linspace(0, window_width, actual_points, dtype=np.int32)
                    for j in range(len(y) - 1):
                        cv2.line(img, (x[j], i * window_height // len(exg_channels) + y[j]),
                                 (x[j + 1], i * window_height // len(exg_channels) + y[j + 1]), (255, 255, 255), 1)
                cv2.imshow(window_name, img)

            cv2.imshow('MediaPipe Hands', image)
            cv2.imshow('Finger Angles', finger_frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    except BaseException as e:
        logging.warning(f'Exception: {e}', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()

if __name__ == '__main__':
    main()
