# playback.py

import os
import time
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Define color constants using ANSI escape codes
COLOR_GREEN = '\033[92m'  # Green
COLOR_GRAY = '\033[90m'   # Gray
COLOR_RESET = '\033[0m'    # Reset color

def clear_console():
    """Clears the terminal console."""
    os.system('cls' if os.name == 'nt' else 'clear')

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
        raise FileNotFoundError("Preprocessing artifacts not found.")

    scaler = joblib.load(scaler_path)
    emg_params = joblib.load(emg_params_path)
    logging.info("Preprocessing parameters loaded successfully.")
    return scaler, emg_params

def extract_emg_features(emg_data, frame_size_samples):
    """
    Extract statistical features from EMG data for a given frame size.
    This function mirrors the feature extraction process used during training.

    Parameters:
    - emg_data: Numpy array of EMG data for the frame.
    - frame_size_samples: Number of samples in the frame.

    Returns:
    - features: List of extracted features.
    """
    features = []
    for channel in range(emg_data.shape[1]):
        channel_data = emg_data[:, channel]
        if len(channel_data) == 0:
            # If no data, append zeros
            features.extend([0, 0, 0, 0, 0, 0, 0])
            continue
        # Time-domain features
        mav = np.mean(np.abs(channel_data))
        rms = np.sqrt(np.mean(channel_data ** 2))
        var = np.var(channel_data)
        std = np.std(channel_data)
        zc = np.sum(np.diff(np.sign(channel_data)) != 0)
        ssc = np.sum(np.diff(np.sign(np.diff(channel_data))) != 0)
        wl = np.sum(np.abs(np.diff(channel_data)))
        features.extend([mav, rms, var, std, zc, ssc, wl])
    return features

def find_nearest_previous_timestamp(emg_timestamps, target_timestamp):
    """
    Find the index of the nearest previous EMG timestamp for a given finger timestamp.
    Uses binary search for efficiency.
    """
    idx = np.searchsorted(emg_timestamps, target_timestamp, side='right') - 1
    if idx < 0:
        return None
    return idx

def create_labels(fingers_angles_df, joint_angle_thresholds):
    """
    Create binary labels indicating whether each finger group is closed.
    Maps the model's three output labels to five individual fingers.

    Parameters:
    - fingers_angles_df: DataFrame containing finger angles with columns:
      ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']
    - joint_angle_thresholds: Dictionary containing threshold angles for each finger.

    Returns:
    - y: Numpy array of shape (num_samples, 3) with binary labels for each finger group.
    """
    y = []
    for _, row in fingers_angles_df.iterrows():
        thumb_closed = int(row['THUMB'] < joint_angle_thresholds['thumb'])
        index_middle_closed = int(
            (row['INDEX'] < joint_angle_thresholds['index']) or
            (row['MIDDLE'] < joint_angle_thresholds['middle'])
        )
        ring_pinky_closed = int(
            (row['RING'] < joint_angle_thresholds['ring']) or
            (row['PINKY'] < joint_angle_thresholds['pinky'])
        )
        label = [thumb_closed, index_middle_closed, ring_pinky_closed]
        y.append(label)
    return np.array(y)

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
        thumb, index_middle, ring_pinky = pred
        # Each group maps to specific fingers
        finger_pred.append([
            thumb,            # Thumb
            index_middle,     # Index
            index_middle,     # Middle
            ring_pinky,       # Ring
            ring_pinky        # Pinky
        ])
    return np.array(finger_pred)

def playback_predictions(y_true, y_pred, playback_rate=0.5):
    """
    Playback the actual vs predicted labels for each frame with visual indicators.

    Parameters:
    - y_true: Numpy array of true labels (shape: [num_samples, 3])
    - y_pred: Numpy array of predicted labels (shape: [num_samples, 3])
    - playback_rate: Float, time in seconds to wait between frames
    """
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    num_samples = y_true.shape[0]

    # Map true and predicted labels to five fingers
    y_true_fingers = map_model_predictions_to_fingers(y_true)
    y_pred_fingers = map_model_predictions_to_fingers(y_pred)

    for i in range(num_samples):
        clear_console()

        actual_labels = y_true_fingers[i]
        predicted_labels = y_pred_fingers[i]

        print(f"Frame {i+1}/{num_samples}\n")
        print("Actual Labels vs Predicted Labels:")
        for idx, finger in enumerate(fingers):
            actual_status = f"{COLOR_GREEN}●{COLOR_RESET}" if actual_labels[idx] else f"{COLOR_GRAY}●{COLOR_RESET}"
            predicted_status = f"{COLOR_GREEN}●{COLOR_RESET}" if predicted_labels[idx] else f"{COLOR_GRAY}●{COLOR_RESET}"
            print(f"{finger}: Actual: {actual_status}  Predicted: {predicted_status}")

        # Wait to simulate real-time processing
        time.sleep(playback_rate)

    print("\nPlayback completed.")

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Paths to data, model, and preprocessing information
    emg_file = 'EMG Hand Data 20241031_001704/emg.csv'  # Update with your EMG data path
    fingers_file = 'EMG Hand Data 20241031_001704/finger_angles.csv'  # Update with your finger angles data path
    model_dir = 'model_20241031_190538'  # Directory where your model and preprocessing artifacts are saved
    model_file = os.path.join(model_dir, 'model.keras')  # Path to your trained model file

    # Configuration Parameters (should match those used during training)
    SAMPLING_RATE = 250  # in Hz
    SELECTED_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]  # EMG channels used during training
    FRAME_SIZES = [1, 2, 4, 8, 16, 32, 48, 64]  # Frame sizes in samples

    # Thresholds for determining if a finger is closed (midpoints)
    JOINT_ANGLE_THRESHOLDS = {
        'thumb':    (132.5  + 157.5)    / 2,    # (132.5 + 157.5) / 2
        'index':    (60     + 160)      / 2,    # (60 + 160) / 2
        'middle':   (40     + 167.5)    / 2,    # (40 + 167.5) / 2
        'ring':     (30     + 167.5)    / 2,    # (30 + 167.5) / 2
        'pinky':    (30     + 167.5)    / 2     # (30 + 167.5) / 2
    }

    # Load the trained model
    if not os.path.exists(model_file):
        logging.error(f"Model file not found at {model_file}.")
        return

    model = load_model(model_file)
    logging.info("Model loaded successfully.")

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
        return

    # Check for required columns in finger data
    expected_finger_columns = ['timestamp', 'THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']
    if not all(col in fingers_df.columns for col in expected_finger_columns):
        logging.error("Finger data file is missing required columns.")
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

        features = []
        for frame_size in FRAME_SIZES:
            start_idx = max(0, nearest_emg_idx - frame_size + 1)
            end_idx = nearest_emg_idx + 1  # Include the nearest sample

            frame_data = emg_data[start_idx:end_idx, :]
            frame_features = extract_emg_features(frame_data, frame_size)
            features.extend(frame_features)

        feature_list.append(features)
        # Collect the corresponding finger angles for label creation
        corresponding_finger_angles.append(fingers_angles.iloc[idx])

    # Convert feature list to numpy array
    X = np.array(feature_list)
    logging.info(f"Feature extraction completed. Shape: {X.shape}")
    logging.info(f"Number of skipped samples: {skipped}")

    if X.size == 0:
        logging.error("No valid samples found after feature extraction.")
        return

    # Convert collected finger angles to DataFrame
    fingers_angles_df = pd.DataFrame(corresponding_finger_angles)

    # Create labels using the create_labels function
    y_true = create_labels(fingers_angles_df, JOINT_ANGLE_THRESHOLDS)
    logging.info(f"Labels created. Shape: {y_true.shape}")

    # Check for any NaNs or infinite values in features
    if np.isnan(X).any() or np.isinf(X).any():
        logging.error("Features contain NaNs or infinite values. Removing such samples.")
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X = X[valid_indices]
        y_true = y_true[valid_indices]
        logging.info(f"After removing invalid samples, feature shape: {X.shape}, label shape: {y_true.shape}")

    # Scale the features using the loaded scaler
    X_scaled = scaler.transform(X)
    logging.info("Feature scaling completed using loaded scaler.")

    # Make predictions
    y_pred_prob = model.predict(X_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)
    logging.info("Predictions made on the new data.")

    # Simulate real-time processing
    playback_rate = 0.01  # seconds per frame (adjust as needed)

    # Start playback
    playback_predictions(y_true, y_pred, playback_rate=playback_rate)

if __name__ == '__main__':
    main()
