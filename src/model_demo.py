# real_time_demo_with_data.py

import os
import json
import pandas as pd
import numpy as np
import time
from data import load_and_preprocess_data, create_aggregated_binary_frames
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

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

def main():
    # Paths to data, model, and preprocessing information
    hand_file = 'fingers_1.csv'  # Update with your actual file path
    emg_file = 'emg_1.csv'        # Update with your actual file path
    preprocessing_file = 'EMG_Model_20241016_221813/preprocessing_info.json'
    model_file = 'EMG_Model_20241016_221813/emg_lstm_model.keras'  # Update with the path to your model

    # Load preprocessing configuration and model
    preprocessing_data = load_preprocessing_data(preprocessing_file)
    model = load_model(model_file)
    scaler_emg = MinMaxScaler()
    scaler_emg.min_ = np.array(preprocessing_data['scaler_emg_min'])
    scaler_emg.scale_ = np.array(preprocessing_data['scaler_emg_scale'])
    thresholds = preprocessing_data['closed_thresholds']
    frame_size_ms = preprocessing_data['frame_size_ms']
    sampling_rate = preprocessing_data['sampling_rate']
    selected_channels = preprocessing_data.get('selected_channels', [1, 2, 3, 4])
    emg_columns = [f'emg_channel_{i}' for i in selected_channels]
    aggregate_method = preprocessing_data.get('aggregate_method', [['pinky', 'ring'], ['index', 'middle'], 'thumb'])

    # Load and preprocess data using data.py functions
    hand_df, emg_df, _ = load_and_preprocess_data(
        hand_file, emg_file
    )

    # Aggregate data to create frames
    X, y = create_aggregated_binary_frames(
        hand_df, emg_df
    )

    # Define a playback rate for 60 frames per second
    playback_rate = 1 / 60.0  # 60 frames per second
    
    predict_y = model.predict(X)
    predict_y = (predict_y > 0.5).astype(int)

    # Process and predict for each frame
    for i in range(len(X)):
        clear_console()
        
        # Get the actual and predicted labels
        actual_label = y[i]
        predicted_label = predict_y[i]

        # Display predictions and actual labels
        pinky_ring_dot_pred = f"{COLOR_GREEN}●{COLOR_RESET}" if predicted_label[0] else f"{COLOR_GRAY}●{COLOR_RESET}"
        pointer_middle_dot_pred = f"{COLOR_GREEN}●{COLOR_RESET}" if predicted_label[1] else f"{COLOR_GRAY}●{COLOR_RESET}"
        thumb_dot_pred = f"{COLOR_GREEN}●{COLOR_RESET}" if predicted_label[2] else f"{COLOR_GRAY}●{COLOR_RESET}"

        pinky_ring_dot_act = f"{COLOR_GREEN}●{COLOR_RESET}" if actual_label[0] else f"{COLOR_GRAY}●{COLOR_RESET}"
        pointer_middle_dot_act = f"{COLOR_GREEN}●{COLOR_RESET}" if actual_label[1] else f"{COLOR_GRAY}●{COLOR_RESET}"
        thumb_dot_act = f"{COLOR_GREEN}●{COLOR_RESET}" if actual_label[2] else f"{COLOR_GRAY}●{COLOR_RESET}"

        print("\nActual Labels vs Predicted Labels:")
        print(f"Pinky/Ring:     Actual: {pinky_ring_dot_act}  Predicted: {pinky_ring_dot_pred}")
        print(f"Pointer/Middle: Actual: {pointer_middle_dot_act}  Predicted: {pointer_middle_dot_pred}")
        print(f"Thumb:          Actual: {thumb_dot_act}  Predicted: {thumb_dot_pred}")

        # Wait to simulate 60 FPS playback
        time.sleep(playback_rate)

    print("\nPlayback completed.")

if __name__ == '__main__':
    main()