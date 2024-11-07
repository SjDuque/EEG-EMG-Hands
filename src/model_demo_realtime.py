import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib # For loading the SVM model
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

def collect_emg_data(inlet, emg_data_list, stop_event, required_samples, num_channels):
    """Collect EMG data in real-time and keep only the last required_samples."""
    while not stop_event.is_set():
        try:
            sample, _ = inlet.pull_sample(timeout=1.0)
            if sample:
                emg_data_list.append(sample[:num_channels])  # Keep only required channels
                # Maintain the list to the last 'required_samples' samples
                if len(emg_data_list) > required_samples:
                    emg_data_list.pop(0)
        except Exception as e:
            print(f"Error in EMG data collection: {e}")

def compute_features(frame):
    """
    Compute features from a frame of shape (time_steps, num_channels).

    Returns:
    - feature_vector: 1D array of features.
    """
    features = []
    # For each channel
    for i in range(frame.shape[1]):
        channel_data = frame[:, i]
        # Compute statistical features
        features.append(np.mean(channel_data))
        features.append(np.std(channel_data))
        features.append(np.min(channel_data))
        features.append(np.max(channel_data))
        features.append(np.median(channel_data))
    return np.array(features)

def main():
    # Directory containing the model and preprocessing info
    model_dir = 'EMG_Model_20241016_173750'  # Replace with your actual folder name

    # File paths
    model_file = os.path.join(model_dir, 'emg_svm_model.joblib')
    preprocessing_file = os.path.join(model_dir, 'preprocessing_info.json')  # Preprocessing JSON file
    scaler_file = os.path.join(model_dir, 'feature_scaler.joblib')

    # Load the saved SVM model
    print("Loading the trained SVM model...")
    svm_model = joblib.load(model_file)
    print("SVM model loaded successfully.")

    # Load preprocessing data
    print("Loading preprocessing data...")
    preprocessing_data = load_preprocessing_data(preprocessing_file)
    scaler_emg = MinMaxScaler()
    scaler_emg.min_ = np.array(preprocessing_data['scaler_emg_min'])
    scaler_emg.scale_ = np.array(preprocessing_data['scaler_emg_scale'])
    frame_size_ms = preprocessing_data['frame_size_ms']
    sampling_rate = preprocessing_data['sampling_rate']
    required_samples = int((frame_size_ms / 1000) * sampling_rate)
    best_thresholds = preprocessing_data['best_thresholds']
    num_channels = len(scaler_emg.scale_)
    print("Preprocessing data loaded.")

    # Load the feature scaler
    print("Loading feature scaler...")
    feature_scaler = joblib.load(scaler_file)
    print("Feature scaler loaded.")

    # Initialize EMG stream
    print("Looking for all available EMG streams...")
    streams = resolve_stream('type', 'EMG')
    if len(streams) == 0:
        print("No EMG streams found.")
        return
    inlet = StreamInlet(streams[0])

    # Set up data collection for the required window size
    emg_data_list = []
    stop_event = threading.Event()
    emg_thread = threading.Thread(
        target=collect_emg_data,
        args=(inlet, emg_data_list, stop_event, required_samples, num_channels)
    )
    emg_thread.start()

    # Main loop for real-time prediction
    try:
        while not stop_event.is_set():
            if len(emg_data_list) < required_samples:
                continue  # Wait until we have enough samples

            # Prepare data for prediction
            X = np.array(emg_data_list[-required_samples:])  # Shape: (required_samples, num_channels)
            # Optional: Ensure X has the correct number of channels
            X = X[:, :num_channels]
            X = scaler_emg.transform(X)

            # Compute features from the EMG data
            feature_vector = compute_features(X)  # Shape: (num_features,)
            feature_vector = feature_vector.reshape(1, -1)  # Reshape for scaler and model input

            # Scale the features
            feature_vector_scaled = feature_scaler.transform(feature_vector)

            # Make predictions using the SVM model
            predictions_decision = svm_model.decision_function(feature_vector_scaled)[0]

            # Apply best thresholds
            predictions_binary = (predictions_decision > best_thresholds).astype(int)

            # Convert predictions to colored dots for display
            pinky_ring_dot = f"{COLOR_GREEN}●{COLOR_RESET}" if predictions_binary[0] else f"{COLOR_GRAY}●{COLOR_RESET}"
            pointer_middle_dot = f"{COLOR_GREEN}●{COLOR_RESET}" if predictions_binary[1] else f"{COLOR_GRAY}●{COLOR_RESET}"
            thumb_dot = f"{COLOR_GREEN}●{COLOR_RESET}" if predictions_binary[2] else f"{COLOR_GRAY}●{COLOR_RESET}"

            # Display the predictions
            clear_console()
            print("\nPredictions:")
            print(f"Pinky/Ring:     {pinky_ring_dot}")
            print(f"Pointer/Middle: {pointer_middle_dot}")
            print(f"Thumb:          {thumb_dot}")

            # Optional small delay
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stop_event.set()
        emg_thread.join()

if __name__ == "__main__":
    main()
