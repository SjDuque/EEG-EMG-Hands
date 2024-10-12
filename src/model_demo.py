import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define the sampling rate of your data (in Hz)
sampling_rate = 250  # Adjust as per your data

# ANSI escape codes for colored output
COLOR_GREEN = '\033[92m'  # Green
COLOR_GRAY = '\033[90m'   # Gray
COLOR_RESET = '\033[0m'   # Reset color

def clear_console():
    """
    Clears the terminal console.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def load_and_preprocess_data(hand_file, emg_file):
    """
    Loads and preprocesses the hand and EMG data.
    """
    # Load the data
    hand_df = pd.read_csv(hand_file)
    emg_df = pd.read_csv(emg_file)

    # Convert timestamps to datetime
    hand_df['timestamp_hand'] = pd.to_datetime(hand_df['timestamp_hand'], unit='s')
    emg_df['timestamp'] = pd.to_datetime(emg_df['timestamp'], unit='s')

    # Merge on nearest timestamps
    merged_df = pd.merge_asof(
        emg_df.sort_values('timestamp'),
        hand_df.sort_values('timestamp_hand'),
        left_on='timestamp',
        right_on='timestamp_hand',
        direction='nearest',
        tolerance=pd.Timedelta(milliseconds=20)  # Adjust tolerance if needed
    )

    # Drop rows where no match was found within the tolerance
    merged_df.dropna(inplace=True)

    # Define columns for normalization
    emg_columns = [col for col in emg_df.columns if 'emg_channel' in col]
    percent_columns = [col for col in hand_df.columns if '_percent' in col]

    # Normalize EMG data
    scaler_emg = MinMaxScaler()
    merged_df[emg_columns] = scaler_emg.fit_transform(merged_df[emg_columns])

    # Normalize percent columns if necessary (assuming between 0 and 100)
    if merged_df[percent_columns].max().max() > 1:
        merged_df[percent_columns] = merged_df[percent_columns] / 100.0

    return merged_df, emg_columns, percent_columns

def create_frames_for_prediction(merged_df, emg_columns, frame_size_ms, sampling_rate):
    """
    Creates sequences (frames) of EMG data for prediction.
    """
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
    # File paths (update these if necessary)
    hand_file = 'hand_data.csv'
    emg_file = 'emg_data.csv'
    model_file = 'emg_lstm_model.h5'  # Ensure this file exists

    # Load the saved model
    print("Loading the trained model...")
    model = load_model(model_file)
    print("Model loaded successfully.")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    merged_df, emg_columns, percent_columns = load_and_preprocess_data(hand_file, emg_file)
    print("Data loaded and preprocessed.")

    # Define thresholds for classification (same as used during training)
    thresholds = {
        'pinky_ring': 0.5,       # Threshold for pinky and ring fingers being down
        'pointer_middle': 0.5,   # Threshold for pointer and middle fingers being down
        'thumb': 0.5             # Threshold for thumb being open
    }

    # Create frames for prediction
    frame_size_ms = 100  # Use the same frame size as during training
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
    # Binarize predictions using 0.5 threshold
    predictions_binary = (predictions > 0.5).astype(int)
    print("Predictions completed.")

    # Iterate through each frame and display the data
    print("Starting real-time display...")
    time.sleep(2)  # Brief pause before starting

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
        # Green dot for active (1), Gray dot for inactive (0)
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

        # Sleep to simulate real-time (based on frame rate)
        time.sleep(frame_size_ms / 1000.0)

    print("\nReal-time display completed.")

if __name__ == "__main__":
    main()
