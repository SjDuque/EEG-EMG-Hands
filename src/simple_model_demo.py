# simple_model_demo.py

import os
import json
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import logging

# Define constants for ANSI color codes for output
COLOR_GREEN = '\033[92m'  # Green
COLOR_GRAY = '\033[90m'   # Gray
COLOR_RESET = '\033[0m'   # Reset color

def clear_console():
    """Clears the terminal console."""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # Paths to data, model, and preprocessing information
    
    emg_file = 'EMG Hand Data 20241016_170438/emg.csv'
    fingers_file = 'EMG Hand Data 20241016_170438/fingers.csv'
    model_file = 'simple_emg_model.keras'  # Path to your trained simple model file

    # Configuration Parameters (should match those used during training)
    SELECTED_CHANNELS = [1, 2, 3, 4, 5]  # EMG channels used during training
    FRAME_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]  # Frame sizes in samples

    # Thresholds for determining if a finger is closed (midpoints)
    JOINT_ANGLE_THRESHOLDS = {
        'thumb': 145.0,
        'index': 95.0,
        'middle': 91.25,
        'ring': 91.25,
        'pinky': 91.25
    }

    # Load the trained model
    model = load_model(model_file)
    logging.info("Model loaded successfully.")

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
    expected_finger_columns = ['thumb_angle', 'index_angle', 'middle_angle', 'ring_angle', 'pinky_angle', 'timestamp']
    if not all(col in fingers_df.columns for col in expected_finger_columns):
        logging.error("Finger data file is missing required columns.")
        return

    # Drop any NaN values in finger data for angle columns
    fingers_df.dropna(subset=expected_finger_columns, inplace=True)
    emg_df.dropna(subset=expected_emg_columns, inplace=True)

    # Extract EMG channels and timestamps
    emg_channels = [f'emg_channel_{ch}' for ch in SELECTED_CHANNELS]
    emg_data = emg_df[emg_channels].values
    emg_timestamps = emg_df['timestamp'].values

    # Extract finger timestamps and angles
    fingers_timestamps = fingers_df['timestamp'].values
    fingers_angles = fingers_df[['thumb_angle', 'index_angle', 'middle_angle', 'ring_angle', 'pinky_angle']]

    # Initialize feature list and labels
    feature_list = []
    corresponding_finger_angles = []
    skipped = 0

    def find_nearest_previous_timestamp(emg_timestamps, target_timestamp):
        indices = np.where(emg_timestamps <= target_timestamp)[0]
        if len(indices) == 0:
            return None
        return indices[-1]

    def extract_emg_features(emg_data, frame_size_samples):
        features = []
        for channel in range(emg_data.shape[1]):
            channel_data = emg_data[:, channel]
            if len(channel_data) == 0:
                # If no data, append zeros
                features.extend([0, 0, 0, 0, 0, 0, 0])
                continue
            mav = np.mean(np.abs(channel_data))
            rms = np.sqrt(np.mean(channel_data ** 2))
            var = np.var(channel_data)
            std = np.std(channel_data)
            zc = np.sum(np.diff(np.sign(channel_data)) != 0)
            ssc = np.sum(np.diff(np.sign(np.diff(channel_data))) != 0)
            wl = np.sum(np.abs(np.diff(channel_data)))
            features.extend([mav, rms, var, std, zc, ssc, wl])
        return features

    # Collect features and labels
    for idx, finger_timestamp in enumerate(fingers_timestamps):
        nearest_emg_idx = find_nearest_previous_timestamp(emg_timestamps, finger_timestamp)
        if nearest_emg_idx is None:
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
        corresponding_finger_angles.append(fingers_angles.iloc[idx])

    # Convert to numpy arrays
    X = np.array(feature_list)
    fingers_angles_df = pd.DataFrame(corresponding_finger_angles)

    # Create labels
    y = []
    for _, row in fingers_angles_df.iterrows():
        label = [
            int(row['thumb_angle'] < JOINT_ANGLE_THRESHOLDS['thumb']),
            int(row['index_angle'] < JOINT_ANGLE_THRESHOLDS['index']),
            int(row['middle_angle'] < JOINT_ANGLE_THRESHOLDS['middle']),
            int(row['ring_angle'] < JOINT_ANGLE_THRESHOLDS['ring']),
            int(row['pinky_angle'] < JOINT_ANGLE_THRESHOLDS['pinky'])
        ]
        y.append(label)
    y = np.array(y)

    # Feature scaling (use the same scaler used during training)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # In practice, use the scaler saved during training
    logging.info("Features scaled.")

    # Simulate real-time processing
    playback_rate = 1 / 15.0  # Simulate 15 frames per second

    # Make predictions
    y_pred_prob = model.predict(X_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Process and predict for each frame
    for i in range(len(X_scaled)):
        clear_console()

        # Get the actual and predicted labels
        actual_label = y[i]
        predicted_label = y_pred[i]

        # Map fingers to indices
        fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

        print("\nActual Labels vs Predicted Labels:")
        for idx, finger in enumerate(fingers):
            actual_status = f"{COLOR_GREEN}●{COLOR_RESET}" if actual_label[idx] else f"{COLOR_GRAY}●{COLOR_RESET}"
            predicted_status = f"{COLOR_GREEN}●{COLOR_RESET}" if predicted_label[idx] else f"{COLOR_GRAY}●{COLOR_RESET}"
            print(f"{finger}: Actual: {actual_status}  Predicted: {predicted_status}")

        # Wait to simulate real-time processing
        time.sleep(playback_rate)

    print("\nPlayback completed.")

if __name__ == '__main__':
    main()
