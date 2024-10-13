import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pylsl
from pylsl import StreamInlet, resolve_stream
import threading
import time

# ANSI escape codes for colored output
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

def collect_emg_data(inlet, emg_data_list, stop_event, required_samples):
    """Collect EMG data in real-time and keep only the last required_samples."""
    while not stop_event.is_set():
        try:
            sample, _ = inlet.pull_sample(timeout=1.0)
            if sample:
                emg_data_list.append(sample[:4])  # Keep only the first 4 channels
                # Maintain the list to the last 'required_samples' samples
                if len(emg_data_list) > required_samples:
                    emg_data_list.pop(0)
        except Exception as e:
            print(f"Error in EMG data collection: {e}")

def main():
    # File paths (update as necessary)
    model_file = 'EMG_Model_20241013_144632/emg_lstm_model.keras'
    preprocessing_file = 'EMG_Model_20241013_144632/preprocessing_info.json'  # Preprocessing JSON file

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
    frame_size_ms = preprocessing_data['frame_size_ms']
    sampling_rate = preprocessing_data['sampling_rate']
    required_samples = int((frame_size_ms / 1000) * sampling_rate)
    print("Preprocessing data loaded.")

    # Initialize EMG stream
    print("Looking for all available EMG streams...")
    streams = resolve_stream('type', 'EMG')
    if len(streams) == 0:
        print("No EMG streams found.")
        return
    inlet = StreamInlet(streams[0])

    # Set up data collection for a 100 ms window (25 samples at 250 Hz)
    emg_data_list = []
    stop_event = threading.Event()
    emg_thread = threading.Thread(target=collect_emg_data, args=(inlet, emg_data_list, stop_event, required_samples))
    emg_thread.start()

    # Main loop for real-time prediction
    try:
        while not stop_event.is_set():
            if len(emg_data_list) < required_samples:
                continue  # Wait until we have enough samples

            # Prepare data for prediction
            X = scaler_emg.transform(emg_data_list[-required_samples:])
            X = np.array([X])  # Reshape for model input
            # Make predictions and debug output
            predictions = model.predict(X, verbose=0)
            # print("Raw Predictions:", predictions)  # Debugging line to see prediction values
            predictions_binary = (predictions > 0.5).astype(int)

            # Convert predictions to colored dots for display
            pinky_ring_dot = f"{COLOR_GREEN}●{COLOR_RESET}" if predictions_binary[0][0] else f"{COLOR_GRAY}●{COLOR_RESET}"
            pointer_middle_dot = f"{COLOR_GREEN}●{COLOR_RESET}" if predictions_binary[0][1] else f"{COLOR_GRAY}●{COLOR_RESET}"
            thumb_dot = f"{COLOR_GREEN}●{COLOR_RESET}" if predictions_binary[0][2] else f"{COLOR_GRAY}●{COLOR_RESET}"

            # Display the predictions
            clear_console()
            print("\nPredictions:")
            print(f"Pinky/Ring:     {pinky_ring_dot}")
            print(f"Pointer/Middle: {pointer_middle_dot}")
            print(f"Thumb:          {thumb_dot}")

            # Optional small delay
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stop_event.set()
        emg_thread.join()

if __name__ == "__main__":
    main()
