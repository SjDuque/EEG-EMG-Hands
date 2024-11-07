# model_playback.py

import os
import time
import logging
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import cv2  # Import OpenCV
import bisect  # Import bisect for efficient searching

from model_utils import (
    find_nearest_previous_timestamp,
    create_labels,
    print_label_distribution,
    build_cnn_model,
    plot_learning_curves
)

# Define color constants using BGR for OpenCV
COLOR_GREEN = (0, 255, 0)    # Green
COLOR_GRAY = (128, 128, 128) # Gray
COLOR_RED = (0, 0, 255)      # Red
COLOR_WHITE = (255, 255, 255) # White

def load_preprocessing_params(model_dir):
    """
    Load the feature scaler and EMG normalization parameters from disk.

    Parameters:
    - model_dir: Directory where the model and preprocessing artifacts are saved.

    Returns:
    - scaler: Loaded StandardScaler object.
    - emg_params: Dictionary containing 'means' and 'stds' for EMG normalization.
    """
    scaler_path = os.path.join(model_dir, 'feature_scaler.joblib')
    emg_params_path = os.path.join(model_dir, 'emg_normalization_params.joblib')

    if not os.path.exists(scaler_path) or not os.path.exists(emg_params_path):
        logging.error("Scaler or EMG normalization parameters not found in the specified model directory.")
        # Log available files for debugging
        if os.path.exists(model_dir):
            logging.error(f"Available files in {model_dir}: {os.listdir(model_dir)}")
        else:
            logging.error(f"Model directory {model_dir} does not exist.")
        raise FileNotFoundError("Preprocessing artifacts not found.")

    scaler = joblib.load(scaler_path)
    emg_params = joblib.load(emg_params_path)
    logging.info("Preprocessing parameters loaded successfully.")
    return scaler, emg_params

def map_model_predictions_to_fingers(y_pred):
    """
    Maps the model's three output labels to five individual fingers.

    Parameters:
    - y_pred: Numpy array of shape (num_samples, 3) with binary predictions.

    Returns:
    - finger_pred: Numpy array of shape (num_samples, 5) with binary predictions for each finger.
    """
    finger_pred = []
    for pred in y_pred:
        thumb, index, middle, ring_pinky = pred
        # Each group maps to specific fingers
        finger_pred.append([
            thumb,            # Thumb
            index,     # Index
            middle,     # Middle
            ring_pinky,       # Ring
            ring_pinky        # Pinky
        ])
    return np.array(finger_pred)

def smooth_predictions(y_pred_prob, beta=0.9):
    """
    Applies exponential moving average smoothing to the prediction probabilities.

    Parameters:
    - y_pred_prob: Numpy array of shape (num_samples, num_classes) with prediction probabilities.
    - beta: Float, smoothing factor between 0 and 1.

    Returns:
    - smoothed_prob: Numpy array of the same shape as y_pred_prob with smoothed probabilities.
    """
    smoothed_prob = np.copy(y_pred_prob)
    for i in range(1, len(y_pred_prob)):
        smoothed_prob[i] = beta * smoothed_prob[i-1] + (1 - beta) * y_pred_prob[i]
    return smoothed_prob

def load_camera_frames(camera_frames_dir):
    """
    Loads and sorts camera frame file paths based on their timestamps extracted from filenames.

    Parameters:
    - camera_frames_dir: Directory containing camera frame images.

    Returns:
    - sorted_camera_timestamps: Sorted list of camera frame timestamps.
    - sorted_camera_filepaths: Sorted list of camera frame filepaths corresponding to the timestamps.
    """
    camera_frames = []
    for filename in os.listdir(camera_frames_dir):
        if filename.lower().endswith(('.jpg', '.png')):
            try:
                # Extract timestamp from filename, e.g., frame_1386389.971473.jpg
                base_name = os.path.splitext(filename)[0]  # Remove extension
                timestamp_str = base_name.split('_')[1]    # Get the part after 'frame_'
                timestamp = float(timestamp_str)
                filepath = os.path.join(camera_frames_dir, filename)
                camera_frames.append((timestamp, filepath))
            except (IndexError, ValueError) as e:
                logging.warning(f"Filename {filename} does not match expected format. Skipping.")
                continue

    # Sort camera frames by timestamp
    sorted_camera_frames = sorted(camera_frames, key=lambda x: x[0])
    sorted_camera_timestamps = [frame[0] for frame in sorted_camera_frames]
    sorted_camera_filepaths = [frame[1] for frame in sorted_camera_frames]
    logging.info(f"Loaded and sorted {len(sorted_camera_frames)} camera frames.")
    return sorted_camera_timestamps, sorted_camera_filepaths

def find_closest_camera_frame(emg_timestamp, camera_timestamps, camera_filepaths, last_idx):
    """
    Finds the filepath of the camera frame closest to the given EMG timestamp using binary search.

    Parameters:
    - emg_timestamp: Float, timestamp of the current EMG frame.
    - camera_timestamps: Sorted list of camera frame timestamps.
    - camera_filepaths: Sorted list of camera frame filepaths.
    - last_idx: Integer, the last searched index to optimize search.

    Returns:
    - filepath: String, filepath of the closest camera frame or None if not found.
    - new_last_idx: Integer, updated last index for optimization.
    """
    idx = bisect.bisect_left(camera_timestamps, emg_timestamp, lo=last_idx)
    if idx == 0:
        return camera_filepaths[0], idx
    if idx == len(camera_timestamps):
        return camera_filepaths[-1], idx
    before = camera_timestamps[idx - 1]
    after = camera_timestamps[idx]
    if abs(emg_timestamp - before) <= abs(after - emg_timestamp):
        return camera_filepaths[idx - 1], idx - 1
    else:
        return camera_filepaths[idx], idx

def overlay_text(frame, text, position, color=COLOR_WHITE, font_scale=0.6, thickness=2):
    """
    Overlays text on a frame.

    Parameters:
    - frame: The image frame.
    - text: The text string to overlay.
    - position: Tuple (x, y) for text position.
    - color: BGR color tuple.
    - font_scale: Scale of the font.
    - thickness: Thickness of the text.

    Returns:
    - frame: The image frame with overlaid text.
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def playback_predictions(y_true, y_pred, emg_timestamps, camera_timestamps=None, camera_filepaths=None, base_delay=1/120, save_video=False, video_path='playback_output.avi'):
    """
    Playback the actual vs predicted labels for each frame with visual indicators.
    Optionally displays corresponding camera frames using OpenCV.
    Allows dynamic adjustment of playback speed.

    Parameters:
    - y_true: Numpy array of true labels (shape: [num_samples, 5])
    - y_pred: Numpy array of predicted labels (shape: [num_samples, 5])
    - emg_timestamps: Numpy array of EMG timestamps for each frame
    - camera_timestamps: Sorted list of camera frame timestamps.
    - camera_filepaths: Sorted list of camera frame filepaths.
    - base_delay: Float, base time in seconds to wait between frames at normal speed
    - save_video: Bool, whether to save the playback as a video file
    - video_path: String, path to save the video file
    """
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    num_samples = y_true.shape[0]
    last_camera_idx = 0  # To keep track of the last matched camera frame

    # Initialize OpenCV window
    cv2.namedWindow('Playback', cv2.WINDOW_NORMAL)
    if camera_timestamps and camera_filepaths:
        cv2.resizeWindow('Playback', 800, 600)
        logging.info("OpenCV window 'Playback' initialized with camera frames.")
    else:
        cv2.resizeWindow('Playback', 800, 200)
        logging.info("OpenCV window 'Playback' initialized without camera frames.")

    # Initialize video writer if saving video
    if save_video:
        if camera_timestamps and camera_filepaths:
            frame_height, frame_width = 600, 800
        else:
            frame_height, frame_width = 200, 800
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 120.0, (frame_width, frame_height))
        logging.info(f"Video will be saved to {video_path}")

    # Initialize playback speed variables
    playback_speed = 1.0  # Normal speed
    min_speed = 0.1        # Minimum speed (10% of normal)
    max_speed = 4.0        # Maximum speed (400% of normal)
    speed_step = 0.1       # Increment/decrement step

    paused = False

    for i in range(num_samples):
        if not paused:
            if camera_timestamps and camera_filepaths:
                emg_timestamp = emg_timestamps[i]
                cam_filepath, last_camera_idx = find_closest_camera_frame(
                    emg_timestamp,
                    camera_timestamps,
                    camera_filepaths,
                    last_camera_idx
                )

                if os.path.exists(cam_filepath):
                    frame = cv2.imread(cam_filepath)
                    if frame is None:
                        logging.warning(f"Failed to load image at {cam_filepath}. Displaying black frame.")
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Black frame
                else:
                    logging.warning(f"Camera frame file {cam_filepath} does not exist. Displaying black frame.")
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Black frame
            else:
                # Create a blank frame if no camera frames are available
                frame = np.zeros((200, 800, 3), dtype=np.uint8)

            # Overlay playback information on the frame
            overlay_text(frame, f"Playback Speed: {playback_speed:.1f}x", (10, 30), COLOR_WHITE, 0.8, 2)
            overlay_text(frame, f"Frame: {i+1}/{num_samples}", (10, 60), COLOR_WHITE, 0.8, 2)
            # Display Actual vs Predicted Labels
            for idx, finger in enumerate(fingers):
                actual_color = COLOR_RED if y_true[i, idx] else COLOR_GREEN
                predicted_color = COLOR_RED if y_pred[i, idx] else COLOR_GREEN
                
                actual_label = "Actual"
                predicted_label = "Predicted"
                
                y_pos = 90 + idx * 30
                overlay_text(frame, finger, (10, y_pos), COLOR_GRAY, 0.7, 2)  # Display finger name in gray
                overlay_text(frame, actual_label, (100, y_pos), actual_color, 0.7, 2)
                overlay_text(frame, predicted_label, (300, y_pos), predicted_color, 0.7, 2)

            # Show the frame
            cv2.imshow('Playback', frame)

            # Write the frame to video if saving
            if save_video:
                # Resize frame to match video dimensions
                if camera_timestamps and camera_filepaths:
                    resized_frame = cv2.resize(frame, (800, 600))
                else:
                    resized_frame = cv2.resize(frame, (800, 200))
                out.write(resized_frame)

        # Calculate delay based on playback_speed
        delay_time = int(base_delay * 1000 / playback_speed)  # Convert to milliseconds

        # Ensure delay_time is within OpenCV's waitKey limits
        delay_time = max(1, min(delay_time, 1000))

        # Wait and capture key input
        key = cv2.waitKey(delay_time) & 0xFF

        if key == ord('q'):
            logging.info("Playback interrupted by user.")
            break
        elif key == ord('f'):
            playback_speed = min(playback_speed + speed_step, max_speed)
            logging.info(f"Increased playback speed to {playback_speed:.1f}x.")
            time.sleep(0.1)  # Brief pause to prevent rapid changes
        elif key == ord('s'):
            playback_speed = max(playback_speed - speed_step, min_speed)
            logging.info(f"Decreased playback speed to {playback_speed:.1f}x.")
            time.sleep(0.1)  # Brief pause to prevent rapid changes
        elif key == ord('p'):
            paused = not paused
            if paused:
                logging.info("Playback paused.")
                # Display paused message
                pause_frame = frame.copy()
                overlay_text(pause_frame, "Playback Paused. Press 'p' to resume.", (10, y_pos + 40), COLOR_RED, 0.8, 2)
                cv2.imshow('Playback', pause_frame)
                if save_video:
                    resized_pause_frame = cv2.resize(pause_frame, (800, 600) if camera_timestamps and camera_filepaths else (800, 200))
                    out.write(resized_pause_frame)
            else:
                logging.info("Playback resumed.")
            time.sleep(0.1)  # Brief pause to prevent rapid toggling

    # Cleanup
    if save_video:
        out.release()
        logging.info(f"Playback video saved to {video_path}")

    cv2.destroyAllWindows()
    logging.info("Playback completed.")

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Configuration Parameters
    SHOW_CAMERA = True  # Set to False if you don't want to display camera frames
    FOLDER_PATH = 'EMG Hand Data 20241031_001704'  # Update with your data folder path
    CAMERA_FRAMES_DIR = f'{FOLDER_PATH}/camera_frames'  # Update with your camera frames directory

    # Paths to data, model, and preprocessing information
    emg_file = f'{FOLDER_PATH}/emg.csv'  # Update with your EMG data path
    fingers_file = f'{FOLDER_PATH}/finger_angles.csv'  # Update with your finger angles data path
    model_dir = 'model_20241102_034210'  # Directory where your model and preprocessing artifacts are saved
    model_file = os.path.join(model_dir, 'model.keras')  # Path to your trained model file

    # Configuration Parameters (should match those used during training)
    SAMPLING_RATE = 250  # in Hz
    SELECTED_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]  # EMG channels used during training
    NUM_FRAMES = 100  # Number of frames to include before the timestamp

    # Thresholds for determining if a finger is closed (midpoints)
    JOINT_ANGLE_THRESHOLDS = {
        'THUMB':    (132.5  + 157.5)    / 2,    # (132.5 + 157.5) / 2
        'INDEX':    (60     + 160)      / 2,    # (60 + 160) / 2
        'MIDDLE':   (40     + 167.5)    / 2,    # (40 + 167.5) / 2
        'RING':     (30     + 167.5)    / 2,    # (30 + 167.5) / 2
        'PINKY':    (30     + 167.5)    / 2     # (30 + 167.5) / 2
    }

    FINGER_GROUPS = {
        'Thumb': ['THUMB'],
        'Index': ['INDEX'],
        'Middle': ['MIDDLE'],
        'Ring_Pinky': ['RING', 'PINKY'],
    }

    # Load the trained model without compiling
    if not os.path.exists(model_file):
        logging.error(f"Model file not found at {model_file}.")
        return

    try:
        model = load_model(model_file, compile=False)
        logging.info("Model loaded successfully (compile=False).")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    # Load preprocessing parameters
    try:
        scaler, emg_params = load_preprocessing_params(model_dir)
    except FileNotFoundError as e:
        logging.error(e)
        return

    # Load EMG data
    try:
        emg_df = pd.read_csv(emg_file)
        fingers_df = pd.read_csv(fingers_file)
        logging.info("Data files loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return

    # Check for required columns in EMG data
    expected_emg_columns = ['timestamp'] + [f'emg_channel_{ch}' for ch in SELECTED_CHANNELS]
    if not all(col in emg_df.columns for col in expected_emg_columns):
        logging.error("EMG data file is missing required columns.")
        logging.error(f"Available columns: {emg_df.columns.tolist()}")
        return

    # Check for required columns in finger data
    expected_finger_columns = ['timestamp', 'THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']  # Adjusted to match FINGER_GROUPS
    if not all(col in fingers_df.columns for col in expected_finger_columns):
        logging.error("Finger data file is missing required columns.")
        logging.error(f"Available columns: {fingers_df.columns.tolist()}")
        return

    # Drop any NaN values in finger data for angle columns
    fingers_df.dropna(subset=expected_finger_columns, inplace=True)
    emg_df.dropna(subset=expected_emg_columns, inplace=True)

    # Normalize EMG data per channel using loaded normalization parameters
    emg_channels = [f'emg_channel_{ch}' for ch in SELECTED_CHANNELS]
    emg_df[emg_channels] = (emg_df[emg_channels] - emg_params['means']) / emg_params['stds']
    logging.info("EMG data normalized using loaded parameters.")

    # Extract EMG channels and timestamps
    emg_data = emg_df[emg_channels].values
    emg_timestamps = emg_df['timestamp'].values
    logging.info("EMG data extracted.")

    # Extract finger timestamps and angles
    fingers_timestamps = fingers_df['timestamp'].values
    fingers_angles = fingers_df[['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']].copy()
    logging.info("Finger data extracted.")

    # Initialize feature list and collect corresponding finger angles
    feature_list = []
    corresponding_finger_angles = []
    skipped = 0

    # Collect features and labels
    for idx, finger_timestamp in enumerate(fingers_timestamps):
        nearest_emg_idx = find_nearest_previous_timestamp(emg_timestamps, finger_timestamp)
        if nearest_emg_idx is None:
            # If no previous EMG data, skip this finger timestamp
            logging.warning(f"No EMG data before finger timestamp {finger_timestamp}. Skipping.")
            skipped += 1
            continue

        # Ensure there are enough frames
        start_idx = nearest_emg_idx - NUM_FRAMES + 1
        end_idx = nearest_emg_idx + 1  # Exclusive

        if start_idx < 0:
            logging.warning(f"Not enough EMG data before timestamp {finger_timestamp}. Skipping.")
            skipped += 1
            continue

        # Extract the sequence of frames
        frame_sequence = emg_data[start_idx:end_idx, :]  # Shape: (NUM_FRAMES, num_channels)
        feature_list.append(frame_sequence)
        corresponding_finger_angles.append(fingers_angles.iloc[idx])

    # Convert feature list to numpy array
    X = np.array(feature_list)  # Shape: (num_samples, NUM_FRAMES, num_channels)
    logging.info(f"Feature extraction completed. Shape: {X.shape}")
    logging.info(f"Number of skipped samples: {skipped}")

    if X.size == 0:
        logging.error("No valid samples found after feature extraction.")
        return

    # Convert collected finger angles to DataFrame
    fingers_angles_df = pd.DataFrame(corresponding_finger_angles)

    # Create labels using the create_labels function
    y_true, group_names = create_labels(fingers_angles_df, FINGER_GROUPS, JOINT_ANGLE_THRESHOLDS)
    logging.info(f"Labels created. Shape: {y_true.shape}")

    # Check for any NaNs or infinite values in features
    if np.isnan(X).any() or np.isinf(X).any():
        logging.error("Features contain NaNs or infinite values. Removing such samples.")
        valid_indices = ~np.isnan(X).any(axis=(1,2)) & ~np.isinf(X).any(axis=(1,2))
        X = X[valid_indices]
        y_true = y_true[valid_indices]
        logging.info(f"After removing invalid samples, feature shape: {X.shape}, label shape: {y_true.shape}")

    # Scale the features using the loaded scaler
    # The scaler was fitted on flattened data during training
    num_samples, num_frames, num_channels = X.shape
    X_reshaped = X.reshape(-1, num_channels)
    X_scaled = scaler.transform(X_reshaped).reshape(num_samples, num_frames, num_channels)
    logging.info("Feature scaling completed using loaded scaler.")

    # Make predictions
    y_pred_prob = model.predict(X_scaled)
    logging.info("Predictions made on the new data.")

    # Apply smoothing to the prediction probabilities
    beta = 0.75
    smoothed_prob = smooth_predictions(y_pred_prob, beta=beta)
    logging.info(f"Applied exponential moving average smoothing with beta={beta}.")

    # Threshold the smoothed probabilities to obtain binary predictions
    y_pred_smoothed = (smoothed_prob > 0.5).astype(int)
    logging.info("Thresholded smoothed probabilities to obtain binary predictions.")

    # Map y_true and y_pred_smoothed to 5-finger labels
    y_true_fingers = map_model_predictions_to_fingers(y_true)
    y_pred_fingers = map_model_predictions_to_fingers(y_pred_smoothed)
    logging.info("Mapped y_true and y_pred to 5-finger labels.")

    # Debugging: Print shapes
    logging.debug(f"y_true shape: {y_true.shape}")
    logging.debug(f"y_pred_smoothed shape: {y_pred_smoothed.shape}")
    logging.debug(f"y_true_fingers shape: {y_true_fingers.shape}")
    logging.debug(f"y_pred_fingers shape: {y_pred_fingers.shape}")

    # Verify that y_true_fingers and y_pred_fingers have 5 labels per sample
    if y_true_fingers.shape[1] != 5 or y_pred_fingers.shape[1] != 5:
        logging.error("Mapped labels do not have 5 classes. Playback aborted.")
        return

    # Load camera frames if required
    camera_timestamps = None
    camera_filepaths = None
    if SHOW_CAMERA:
        if not os.path.exists(CAMERA_FRAMES_DIR):
            logging.error(f"Camera frames directory {CAMERA_FRAMES_DIR} does not exist.")
            return
        camera_timestamps, camera_filepaths = load_camera_frames(CAMERA_FRAMES_DIR)
        if not camera_timestamps:
            logging.warning("No valid camera frames loaded. Camera display will be skipped.")
            SHOW_CAMERA = False

    # Simulate real-time processing

    # Align EMG timestamps with labels
    aligned_emg_timestamps = fingers_timestamps[:y_true_fingers.shape[0]]

    # Verify timestamp ranges
    if SHOW_CAMERA:
        min_emg_timestamp = np.min(aligned_emg_timestamps)
        max_emg_timestamp = np.max(aligned_emg_timestamps)
        min_cam_timestamp = camera_timestamps[0]
        max_cam_timestamp = camera_timestamps[-1]
        logging.info(f"EMG Timestamp Range: {min_emg_timestamp} to {max_emg_timestamp}")
        logging.info(f"Camera Timestamp Range: {min_cam_timestamp} to {max_cam_timestamp}")

        if min_emg_timestamp < min_cam_timestamp or max_emg_timestamp > max_cam_timestamp:
            logging.warning("EMG timestamps extend beyond the range of camera frame timestamps.")
            logging.warning("Some EMG frames may not have corresponding camera frames.")

    # Start playback using smoothed predictions
    playback_predictions(
        y_true_fingers,
        y_pred_fingers,
        emg_timestamps=aligned_emg_timestamps,
        camera_timestamps=camera_timestamps,
        camera_filepaths=camera_filepaths,
        base_delay=1/120,  # Base delay in seconds (8.33 ms)
        save_video=False,   # Set to True if you want to save the playback as a video
        video_path='playback_output.avi'  # Path to save the video file
    )

if __name__ == '__main__':
    main()
