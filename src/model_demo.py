import os
import json
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define constants for ANSI color codes for output
COLOR_GREEN = '\033[92m'  # Green
COLOR_GRAY = '\033[90m'   # Gray
COLOR_RESET = '\033[0m'   # Reset color

def clear_console():
    """Clears the terminal console."""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_preprocessing_data(preprocessing_file):
    """Loads preprocessing data from a JSON file."""
    with open(preprocessing_file, 'r') as f:
        preprocessing_data = json.load(f)
    return preprocessing_data

def load_and_preprocess_data(hand_file, emg_file, scaler_emg, thresholds):
    """Loads and preprocesses the hand and EMG data using provided scaling parameters."""
    # Load the data
    hand_df = pd.read_csv(hand_file)
    emg_df = pd.read_csv(emg_file)

    # Convert timestamps to datetime
    hand_df['timestamp'] = pd.to_datetime(hand_df['timestamp'], unit='s')
    emg_df['timestamp'] = pd.to_datetime(emg_df['timestamp'], unit='s')
    
    # Filter EMG channels [1, 2, 3, 4]
    selected_channels = ['emg_channel_1', 'emg_channel_2', 'emg_channel_3', 'emg_channel_4']
    emg_df = emg_df[['timestamp'] + selected_channels]
    
    # Merge on nearest timestamps
    merged_df = pd.merge_asof(
        emg_df.sort_values('timestamp'),
        hand_df.sort_values('timestamp'),
        left_on='timestamp',
        right_on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(milliseconds=20)  # Adjust tolerance if needed
    )

    # Drop rows where no match was found within the tolerance
    merged_df.dropna(inplace=True)

    # Define EMG and percent columns
    emg_columns = [col for col in emg_df.columns if 'emg_channel' in col]
    percent_columns = [col for col in hand_df.columns if '_percent' in col]

    # Apply scaler to EMG data using loaded scaling parameters
    merged_df[emg_columns] = scaler_emg.transform(merged_df[emg_columns])

    # Normalize percent columns if necessary
    if merged_df[percent_columns].max().max() > 1:
        merged_df[percent_columns] = merged_df[percent_columns] / 100.0

    return merged_df, emg_columns, percent_columns

def create_frames_for_prediction(merged_df, emg_columns, frame_size_ms, sampling_rate):
    """Creates sequences (frames) of EMG data for prediction."""
    frame_size_samples = int((frame_size_ms / 1000) * sampling_rate)
    X = merged_df[emg_columns].values

    # Create sequences
    X_frames = []
    for i in range(len(X) - frame_size_samples + 1):
        X_frames.append(X[i:i + frame_size_samples])

    # Get finger percentages corresponding to each frame
    finger_percentages = merged_df[[col for col in merged_df.columns if '_percent' in col]].iloc[frame_size_samples - 1:].reset_index(drop=True)

    return np.array(X_frames), finger_percentages

def main():
    # File paths (update these as necessary)
    hand_file = 'EMG Hand Data 20241013_142748/fingers.csv'  # Update with your actual file path
    emg_file = 'EMG Hand Data 20241013_142748/emg.csv'        # Update with your actual file path
    model_file = 'EMG_Model_20241013_142814/emg_lstm_model.keras'
    preprocessing_file = 'EMG_Model_20241013_142814/preprocessing_info.json'  # Preprocessing JSON file

    # Load the saved model
    print("Loading the trained model...")
    model = load_model(model_file)
    print("Model loaded successfully.")

    # Load preprocessing data
    print("Loading preprocessing data...")
    preprocessing_data = load_preprocessing_data(preprocessing_file)
    scaler_emg = MinMaxScaler()
    scaler_emg.min_ = np.array(preprocessing_data['scaler_emg_min'])
    scaler_emg.scale_ = np.array(preprocessing_data['scaler_emg_scale'])
    thresholds = preprocessing_data['thresholds']
    frame_size_ms = preprocessing_data['frame_size_ms']
    sampling_rate = preprocessing_data['sampling_rate']
    print("Preprocessing data loaded.")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    merged_df, emg_columns, percent_columns = load_and_preprocess_data(
        hand_file, emg_file, scaler_emg, thresholds
    )
    print("Data loaded and preprocessed.")

    # Create frames for prediction
    X_frames, finger_percentages = create_frames_for_prediction(
        merged_df, emg_columns, frame_size_ms, sampling_rate
    )
    print(f"Total frames for prediction: {len(X_frames)}")

    # Reshape data to fit the model
    X_frames = X_frames.reshape(
        (X_frames.shape[0], X_frames.shape[1], len(emg_columns))
    )

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_frames, verbose=0)
    predictions_binary = (predictions > 0.5).astype(int)
    print("Predictions completed.")

    # Display real-time simulation
    print("Starting real-time display...")
    time.sleep(2)

    for i in range(len(X_frames)):
        clear_console()

        # Get current finger percentages
        current_fingers = finger_percentages.iloc[i]
        pinky_percent = current_fingers['pinky_percent']
        ring_percent = current_fingers['ring_percent']
        index_percent = current_fingers['index_percent']
        middle_percent = current_fingers['middle_percent']
        thumb_percent = current_fingers['thumb_percent']

        # Get current predictions
        pinky_ring_pred = predictions_binary[i][0]
        pointer_middle_pred = predictions_binary[i][1]
        thumb_pred = predictions_binary[i][2]

        # Convert predictions to colored dots
        pinky_ring_dot = f"{COLOR_GREEN}●{COLOR_RESET}" if pinky_ring_pred else f"{COLOR_GRAY}●{COLOR_RESET}"
        pointer_middle_dot = f"{COLOR_GREEN}●{COLOR_RESET}" if pointer_middle_pred else f"{COLOR_GRAY}●{COLOR_RESET}"
        thumb_dot = f"{COLOR_GREEN}●{COLOR_RESET}" if thumb_pred else f"{COLOR_GRAY}●{COLOR_RESET}"

        # Display the information
        print(f"Finger Percentages:")
        print(f"Pinky:   {pinky_percent*100:.1f}%")
        print(f"Ring:    {ring_percent*100:.1f}%")
        print(f"Index:   {index_percent*100:.1f}%")
        print(f"Middle:  {middle_percent*100:.1f}%")
        print(f"Thumb:   {thumb_percent*100:.1f}%")
        print("\nPredictions:")
        print(f"Pinky/Ring:     {pinky_ring_dot}")
        print(f"Pointer/Middle: {pointer_middle_dot}")
        print(f"Thumb:          {thumb_dot}")

        # Sleep to simulate real-time display
        time.sleep(10 / 1000.0)

    print("\nReal-time display completed.")

if __name__ == "__main__":
    main()
