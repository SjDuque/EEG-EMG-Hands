import cv2
import mediapipe as mp
import sys
import numpy as np
import pandas as pd
import time
import pylsl
from pylsl import StreamInlet, resolve_stream, proc_ALL
import threading
import datetime
import os
import shutil
import glob
from tqdm import tqdm  # Import tqdm for the progress bar

# ignore tensorflow related warnings
import warnings
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
# Suppress specific Protobuf deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

def collect_emg_data(inlet, emg_data_list, emg_channel_count, stop_event):
    """Collect EMG data at high frequency using LSL timestamps."""
    while not stop_event.is_set():
        try:
            # Pull a sample from the inlet
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample is None:
                continue

            # Create data row with LSL timestamp and EMG data
            data_row = {'timestamp': timestamp}
            for i in range(emg_channel_count):
                data_row[f'emg_channel_{i + 1}'] = sample[i]
            emg_data_list.append(data_row)

        except Exception as e:
            print(f"Error in EMG data collection: {e}")
            continue

def capture_camera_frames(frames_folder, stop_event, refresh_rate=None, resolution=None, use_half=False):
    """
    Capture camera frames in a separate thread with specified resolution.
    If use_half is True, only the left half of the frame is processed and saved.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        stop_event.set()
        return

    # Set camera resolution if specified
    if resolution is not None:
        width, height = resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Set camera frame rate if specified
    if refresh_rate is not None:
        cap.set(cv2.CAP_PROP_FPS, refresh_rate)

    loop_count = 0
    last_print_time = time.time()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # If use_half is True, crop the frame to the left half
        if use_half:
            height, width = frame.shape[:2]
            frame = frame[:, :width // 2]

        # Get LSL timestamp
        timestamp = pylsl.local_clock()

        # Define filenames
        temp_filename = os.path.join(frames_folder, f"temp_frame_{timestamp:.6f}.jpg")
        final_filename = os.path.join(frames_folder, f"frame_{timestamp:.6f}.jpg")
        latest_temp = os.path.join(frames_folder, "temp_latest_frame.jpg")
        latest_final = os.path.join(frames_folder, "latest_frame.jpg")

        # Save the frame to a temporary file first
        try:
            # Using JPEG format for faster write operations
            cv2.imwrite(temp_filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            # Rename the temporary file to the final filename atomically
            os.rename(temp_filename, final_filename)

            # Update the latest_frame.jpg atomically
            cv2.imwrite(latest_temp, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            os.rename(latest_temp, latest_final)
        except Exception as e:
            print(f"Error saving frame {timestamp}: {e}")

        # Refresh rate log
        loop_count += 1
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            refresh_rate_actual = loop_count / (current_time - last_print_time)
            print(f"Camera Capture Refresh Rate: {refresh_rate_actual:.2f} Hz")
            loop_count = 0
            last_print_time = current_time

    cap.release()

def process_frames(frames_folder, output_csv):
    """Process saved frames with MediaPipe to extract finger angles and save to CSV."""
    mp_hands = mp.solutions.hands
    # mp_drawing = mp.solutions.drawing_utils

    hand_data_list = []

    # Get all frame filenames
    frame_files = sorted(glob.glob(os.path.join(frames_folder, 'frame_*.jpg')))

    total_frames = len(frame_files)
    if total_frames == 0:
        print("No frames to process.")
        return

    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        # Initialize tqdm progress bar
        with tqdm(total=total_frames, desc="Processing Frames", unit="frame") as pbar:
            for frame_file in frame_files:
                # Extract timestamp from filename
                filename = os.path.basename(frame_file)
                timestamp_str = filename[len('frame_'):-len('.jpg')]
                try:
                    timestamp = float(timestamp_str)
                except ValueError:
                    print(f"Invalid timestamp in filename: {filename}")
                    continue

                # Read the image
                image = cv2.imread(frame_file)
                if image is None:
                    print(f"Failed to read image: {frame_file}")
                    # Fill with NaNs if image read fails
                    data_row = {'timestamp': timestamp}
                    for idx in range(21):
                        data_row[f'landmark_{idx}_x'] = np.nan
                        data_row[f'landmark_{idx}_y'] = np.nan
                        data_row[f'landmark_{idx}_z'] = np.nan
                    hand_data_list.append(data_row)
                    pbar.update(1)
                    continue

                # Convert the BGR image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Process the image and detect hands
                results = hands.process(image_rgb)

                # Collect hand tracking data
                data_row = {'timestamp': timestamp}

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Record the landmark positions
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        data_row[f'landmark_{idx}_x'] = landmark.x
                        data_row[f'landmark_{idx}_y'] = landmark.y
                        data_row[f'landmark_{idx}_z'] = landmark.z

                else:
                    # No hand detected; fill with NaNs
                    for idx in range(21):
                        data_row[f'landmark_{idx}_x'] = np.nan
                        data_row[f'landmark_{idx}_y'] = np.nan
                        data_row[f'landmark_{idx}_z'] = np.nan

                hand_data_list.append(data_row)
                pbar.update(1)  # Update the progress bar

    # Save to CSV
    hand_df = pd.DataFrame(hand_data_list)
    hand_df.to_csv(output_csv, index=False)

def main():
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
        emg_channel_count = inlet.info().channel_count()

    # Create data lists
    emg_data_list = []

    # Create a stop event for threads
    stop_event = threading.Event()

    # Start EMG data collection in a separate thread
    if inlet:
        emg_thread = threading.Thread(target=collect_emg_data, args=(inlet, emg_data_list, emg_channel_count, stop_event))
        emg_thread.start()

    # Timestamped folder name
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"EMG Hand Data {now}"
    os.makedirs(folder_name, exist_ok=True)
    frames_folder = os.path.join(folder_name, 'camera_frames')
    os.makedirs(frames_folder, exist_ok=True)
    
    # Specify whether to delete the camera frames folder after processing
    delete_frames = False

    # Start camera capture in a separate thread
    # Optional: Specify desired camera refresh rate (in Hz), or set to None for maximum rate
    camera_refresh_rate = 120  # Set to 120 Hz as desired

    # Optional: Specify desired camera resolution (width, height)
    # If use_half is True, consider setting the width to half of desired to optimize performance
    camera_resolution = (1280, 480)  # e.g., (640, 480), (1280, 720), (1920, 1080), or None

    # Set the use_half parameter here
    use_half = True  # Set to True to use only the left half of the image, False to use the full image

    camera_thread = threading.Thread(
        target=capture_camera_frames,
        args=(frames_folder, stop_event, camera_refresh_rate, camera_resolution, use_half)
    )
    camera_thread.start()

    try:
        # Initialize variables for frame display and framerate calculation
        display_framerate = 0
        frame_update_count = 0
        last_display_time = time.time()
        displayed_frame = None
        latest_frame_path = os.path.join(frames_folder, 'latest_frame.jpg')
        start = time.time()

        while not stop_event.is_set():
            if not os.path.exists(latest_frame_path):
                time.sleep(0.005)
                continue

            # Get the last modification time of the latest_frame.jpg
            try:
                current_mod_time = os.path.getmtime(latest_frame_path)
            except OSError as e:
                print(f"Error accessing {latest_frame_path}: {e}")
                time.sleep(0.005)
                continue

            # Check if the file has been updated since last display
            if displayed_frame != current_mod_time:
                try:
                    # Read and display the latest frame
                    image = cv2.imread(latest_frame_path)
                    if image is not None:
                        cv2.imshow('Camera', image)
                        displayed_frame = current_mod_time
                        frame_update_count += 1
                except cv2.error as e:
                    print(f"libpng error while reading {latest_frame_path}: {e}")

            # Calculate display framerate every second
            current_time = time.time()
            if current_time - last_display_time >= 1.0:
                display_framerate = frame_update_count / (current_time - last_display_time)
                print(f"Display Refresh Rate: {display_framerate:.2f} Hz")
                frame_update_count = 0
                last_display_time = current_time

            # Handle keypress
            if cv2.waitKey(1) & 0xFF == 27:  # Escape key
                stop_event.set()
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        stop_event.set()
    finally:
        # Wait for threads to finish
        if inlet:
            emg_thread.join()
        camera_thread.join()
        cv2.destroyAllWindows()
        
        # Display total elapsed time
        print(f"Elapsed Time: {time.time() - start:.2f} s")
        
        # Display total emg samples
        print(f"Total EMG Samples: {len(emg_data_list)}")

        # Save EMG data
        if emg_data_list:
            emg_df = pd.DataFrame(emg_data_list)
            emg_csv = os.path.join(folder_name, 'emg.csv')
            emg_df.to_csv(emg_csv, index=False)
            print(f"\nEMG data saved to '{emg_csv}'")
        else:
            print("\nNo EMG data to save.")

        # Process the frames to get angles
        fingers_csv = os.path.join(folder_name, 'fingers.csv')
        process_frames(frames_folder, fingers_csv)
        print(f"Finger data saved to '{fingers_csv}'")
        
        
        if delete_frames:
            # Delete the camera frames folder
            try:
                shutil.rmtree(frames_folder)
                print(f"Deleted frames folder '{frames_folder}'")
            except Exception as e:
                print(f"Error deleting frames folder '{frames_folder}': {e}")

        print(f"All data saved to folder '{folder_name}'")

if __name__ == "__main__":
    main()
