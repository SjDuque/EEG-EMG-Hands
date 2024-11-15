import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import pylsl
from pylsl import StreamInlet, resolve_stream
import threading
import datetime
import os
import shutil
from tqdm import tqdm  # For progress bar
import glob  # For processing frames
from random import shuffle

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# ================================
# Configuration Parameters
# ================================
status_labels = ['thumb', 'index', 'middle', 'ring', 'pinky']
# status_lists = [
#     [True, False, True, False, True],
#     [False, True, False, True, False],
#     [True, True, False, False, True],
#     [False, False, True, True, False],
#     [True, False, False, True, True],
# ]
# status_list should be a list of 5 booleans, each representing the status of a finger.
# True means the finger is extended, False means the finger is not extended.
# Create every possible combination of 5 fingers
status_lists = []
for i in range(2 ** (len(status_labels))):
    status_list = [bool((i >> j) & 1) for j in range(5)]
    status_lists.append(status_list)
shuffle(status_lists)  # Randomize the order of status lists
status_switch_interval = 5  # seconds
process_after_recording = True
delete_after_processing = False
save_all_frames = True
use_half = True
camera_refresh_rate = 120  # Hz
camera_resolution = (1280, 480)

# ================================
# Global Variables
# ================================
latest_frame = None
latest_frame_lock = threading.Lock()
current_status = status_lists[0] if status_lists else [False] * 5
status_index = 0
last_status_switch_time = pylsl.local_clock()

# ================================
# Functions
# ================================
def draw_status_circles(image, status):
    """
    Draws five circles on the image.
    Each circle is green if status[i] is True, else red.
    """
    radius = 20
    spacing = 30
    start_x = 50
    start_y = 50
    for i in range(5):
        center_x = start_x + i * (2 * radius + spacing)
        center_y = start_y
        color = (0, 255, 0) if status[i] else (0, 0, 255)  # Green or Red
        cv2.circle(image, (center_x, center_y), radius, color, -1)

def collect_emg_data(inlet, emg_data_list, emg_channel_count, stop_event):
    """Collect EMG data at high frequency using LSL timestamps."""
    sample_counter = 0  # Added for EMG Refresh Rate
    last_time = pylsl.local_clock()  # Added for EMG Refresh Rate
    
    while not stop_event.is_set():
        try:
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                data_row = {'timestamp': timestamp}
                data_row['timestamp_local'] = pylsl.local_clock()
                for i in range(emg_channel_count):
                    data_row[f'emg_channel_{i + 1}'] = sample[i]
                emg_data_list.append(data_row)
                
                # Update EMG sample counter
                sample_counter += 1  # Added for EMG Refresh Rate
                current_time = pylsl.local_clock()  # Added for EMG Refresh Rate
                if current_time - last_time >= 1.0:  # Added for EMG Refresh Rate
                    print(f"EMG Data Rate: {sample_counter} samples/s")  # Added for EMG Refresh Rate
                    sample_counter = 0  # Reset counter
                    last_time = current_time  # Reset timer
        except Exception as e:
            print(f"EMG Collection Error: {e}")

def capture_camera_frames(frames_folder, stop_event, refresh_rate, resolution, use_half, save_all_frames):
    """
    Capture camera frames in a separate thread with specified resolution.
    If use_half is True, only the left half of the frame is processed.
    Frames are saved to disk as 'frame_{timestamp}.jpg' if save_all_frames is True.
    The latest frame is stored in memory for real-time display.
    """
    global latest_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        stop_event.set()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, refresh_rate)

    frame_counter = 0  # Added for Camera Refresh Rate
    last_time = pylsl.local_clock()  # Added for Camera Refresh Rate

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        if use_half:
            frame = frame[:, :frame.shape[1] // 2]

        timestamp = pylsl.local_clock()
        frame_filename = os.path.join(frames_folder, f"frame_{timestamp:.6f}.jpg")
        if save_all_frames:
            try:
                cv2.imwrite(frame_filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            except Exception as e:
                print(f"Error saving frame {timestamp}: {e}")

        with latest_frame_lock:
            latest_frame = frame.copy()

        # Update Camera frame counter
        frame_counter += 1  # Added for Camera Refresh Rate
        current_time = pylsl.local_clock()  # Added for Camera Refresh Rate
        if current_time - last_time >= 1.0:  # Added for Camera Refresh Rate
            print(f"Camera Capture Rate: {frame_counter} Hz")  # Added for Camera Refresh Rate
            frame_counter = 0  # Reset counter
            last_time = current_time  # Reset timer

    cap.release()

def process_frames(frames_folder, output_csv):
    """Process saved frames with MediaPipe to extract finger angles and save to CSV."""
    mp_hands = mp.solutions.hands
    hand_data_list = []
    frame_files = sorted(glob.glob(os.path.join(frames_folder, 'frame_*.jpg')))
    total_frames = len(frame_files)
    if total_frames == 0:
        print("No frames to process.")
        return

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        with tqdm(total=total_frames, desc="Processing Frames") as pbar:
            for frame_file in frame_files:
                filename = os.path.basename(frame_file)
                timestamp_str = filename[len('frame_'):-len('.jpg')]
                try:
                    timestamp = float(timestamp_str)
                except ValueError:
                    print(f"Invalid timestamp in filename: {filename}")
                    continue

                image = cv2.imread(frame_file)
                if image is None:
                    print(f"Failed to read {frame_file}")
                    data_row = {'timestamp': timestamp}
                    for idx in range(21):
                        data_row[f'landmark_{idx}_x'] = np.nan
                        data_row[f'landmark_{idx}_y'] = np.nan
                        data_row[f'landmark_{idx}_z'] = np.nan
                    hand_data_list.append(data_row)
                    pbar.update(1)
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                data_row = {'timestamp': timestamp}

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        data_row[f'landmark_{idx}_x'] = landmark.x
                        data_row[f'landmark_{idx}_y'] = landmark.y
                        data_row[f'landmark_{idx}_z'] = landmark.z
                else:
                    for idx in range(21):
                        data_row[f'landmark_{idx}_x'] = np.nan
                        data_row[f'landmark_{idx}_y'] = np.nan
                        data_row[f'landmark_{idx}_z'] = np.nan

                hand_data_list.append(data_row)
                pbar.update(1)

    if hand_data_list:
        hand_df = pd.DataFrame(hand_data_list)
        hand_df.to_csv(output_csv, index=False)
        print(f"Finger data saved to '{output_csv}'")
    else:
        print("No hand data to save.")

def main():
    global current_status, status_index, last_status_switch_time

    # Initialize EMG stream
    print("Searching for EMG streams...")
    streams = resolve_stream('type', 'EMG')

    if not streams:
        print("No EMG streams found.")
        inlet = None
        emg_channel_count = 0
    else:
        print(f"Found {len(streams)} EMG stream(s).")
        inlet = StreamInlet(streams[0], processing_flags=pylsl.proc_ALL)
        emg_channel_count = inlet.info().channel_count()

    emg_data_list = []
    stop_event = threading.Event()

    if inlet:
        emg_thread = threading.Thread(target=collect_emg_data, args=(inlet, emg_data_list, emg_channel_count, stop_event))
        emg_thread.start()

    # Create directories
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"EMG_Hand_Data_{now}"
    os.makedirs(folder_name, exist_ok=True)
    frames_folder = os.path.join(folder_name, 'camera_frames')
    os.makedirs(frames_folder, exist_ok=True)

    # Start camera capture
    camera_thread = threading.Thread(
        target=capture_camera_frames,
        args=(frames_folder, stop_event, camera_refresh_rate, camera_resolution, use_half, save_all_frames)
    )
    camera_thread.start()

    # Initialize Display Refresh Rate Tracking
    display_counter = 0  # Added for Display Refresh Rate
    last_display_time = pylsl.local_clock()  # Added for Display Refresh Rate

    try:
        while not stop_event.is_set():
            # Get the latest frame from memory
            with latest_frame_lock:
                if latest_frame is not None:
                    display_image = latest_frame.copy()
                else:
                    display_image = None
            
            # Status switching
            current_time = pylsl.local_clock()
            if current_time - last_status_switch_time >= status_switch_interval:
                if status_index % len(status_lists) == 0:
                    shuffle(status_lists)
                current_status = status_lists[status_index % len(status_lists)]
                status_index += 1
                print(f"Switched to status list index {status_index % len(status_lists)}")
                last_status_switch_time = current_time

            if display_image is not None:
                draw_status_circles(display_image, current_status)
                cv2.imshow('Camera', display_image)
                display_counter += 1  # Added for Display Refresh Rate

            # Calculate Display Refresh Rate every second
            if current_time - last_display_time >= 1.0:
                print(f"Display Refresh Rate: {display_counter} Hz")  # Added for Display Refresh Rate
                display_counter = 0  # Reset counter
                last_display_time = current_time  # Reset timer

            # Handle keypress
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                stop_event.set()
                break

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        camera_thread.join()
        if inlet:
            emg_thread.join()
        cv2.destroyAllWindows()

        # Save EMG data
        if emg_data_list:
            emg_df = pd.DataFrame(emg_data_list)
            emg_csv = os.path.join(folder_name, 'emg.csv')
            emg_df.to_csv(emg_csv, index=False)
            print(f"EMG data saved to '{emg_csv}'")
        else:
            print("No EMG data to save.")

        # Process frames
        if process_after_recording and save_all_frames:
            print("Processing saved frames...")
            fingers_csv = os.path.join(folder_name, 'fingers.csv')
            process_frames(frames_folder, fingers_csv)
            if delete_after_processing:
                shutil.rmtree(frames_folder)
                print(f"Deleted frames folder '{frames_folder}'")

        print(f"All data saved to folder '{folder_name}'")

if __name__ == "__main__":
    main()
