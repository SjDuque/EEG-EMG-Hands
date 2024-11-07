import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from pylsl import StreamInlet, resolve_stream
import threading
import time
import glob
from collections import deque  # To store past predictions for smoothing

# ANSI escape codes for colored output
COLOR_GREEN = '\033[92m'  # Green
COLOR_GRAY = '\033[90m'   # Gray
COLOR_RESET = '\033[0m'   # Reset color

def clear_console():
    """Clears the terminal console."""
    os.system('cls' if os.name == 'nt' else 'clear')

def extract_emg_features(emg_data):
    """
    Extract statistical features from EMG data.
    emg_data: numpy array of shape (n_samples, n_channels)
    Returns a feature vector.
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

def collect_emg_data(inlet, emg_data_list, stop_event, max_frame_size, num_channels, selected_channels):
    """Collect EMG data in real-time and keep only the last max_frame_size samples."""
    while not stop_event.is_set():
        try:
            sample, _ = inlet.pull_sample(timeout=1.0)
            if sample:
                # Adjust indices if necessary
                # If the LSL stream provides channels in a different order, adjust this accordingly
                sample = [sample[ch - 1] for ch in selected_channels]
                emg_data_list.append(sample)  # Keep only required channels
                # Maintain the list to the last 'max_frame_size' samples
                if len(emg_data_list) > max_frame_size:
                    emg_data_list.pop(0)
        except Exception as e:
            print(f"Error in EMG data collection: {e}")

def smooth_predictions(predictions, smoothing_buffer, smoothing_window_size):
    """
    Smooth the predictions using a moving average.
    predictions: current binary predictions array (shape: 3, corresponding to 3 finger groups)
    smoothing_buffer: deque buffer storing past predictions
    smoothing_window_size: number of past predictions to consider for smoothing
    Returns smoothed predictions (averaged across the window size).
    """
    smoothing_buffer.append(predictions)
    if len(smoothing_buffer) > smoothing_window_size:
        smoothing_buffer.popleft()

    # Calculate the moving average for each output
    smoothed_preds = np.mean(smoothing_buffer, axis=0)
    return (smoothed_preds > 0.5).astype(int)  # Convert back to binary (0 or 1)

def main():
    # Parameters
    SELECTED_CHANNELS = [1, 2, 3, 4, 5]  # EMG channels to use (adjust based on your setup)
    SAMPLING_RATE = 250  # in Hz
    FRAME_SIZES = [1, 2, 4, 8, 16, 32, 48, 64]  # Frame sizes in samples
    MAX_FRAME_SIZE = max(FRAME_SIZES)
    num_channels = len(SELECTED_CHANNELS)

    SMOOTHING_WINDOW_SIZE = 10  # Adjust this value based on how much smoothing you need

    # Buffer to store past predictions for smoothing
    smoothing_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE)

    # Find the latest model directory
    model_dirs = sorted(glob.glob('model_*'))
    if not model_dirs:
        print("No model directories found.")
        return
    latest_model_dir = model_dirs[-1]
    print(f"Loading model and preprocessing data from {latest_model_dir}...")

    # Load the trained model
    model_path = os.path.join(latest_model_dir, 'model.keras')
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    print("Loading the trained neural network model...")
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Load the feature scaler
    feature_scaler_path = os.path.join(latest_model_dir, 'feature_scaler.joblib')
    if not os.path.exists(feature_scaler_path):
        print(f"Feature scaler file not found at {feature_scaler_path}")
        return
    print("Loading the feature scaler...")
    feature_scaler = joblib.load(feature_scaler_path)
    print("Feature scaler loaded.")

    # Load the EMG normalization parameters
    emg_norm_params_path = os.path.join(latest_model_dir, 'emg_normalization_params.joblib')
    if not os.path.exists(emg_norm_params_path):
        print(f"EMG normalization parameters file not found at {emg_norm_params_path}")
        return
    print("Loading the EMG normalization parameters...")
    emg_norm_params = joblib.load(emg_norm_params_path)
    emg_means = emg_norm_params['means'].values
    emg_stds = emg_norm_params['stds'].values
    print("EMG normalization parameters loaded.")

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
        args=(inlet, emg_data_list, stop_event, MAX_FRAME_SIZE, num_channels, SELECTED_CHANNELS)
    )
    emg_thread.start()

    # Main loop for real-time prediction
    try:
        while not stop_event.is_set():
            if len(emg_data_list) < MAX_FRAME_SIZE:
                continue  # Wait until we have enough samples

            # Prepare data for prediction
            features = []
            for frame_size in FRAME_SIZES:
                frame_data = np.array(emg_data_list[-frame_size:])  # Shape: (frame_size, num_channels)
                # Normalize EMG data per channel using emg_means and emg_stds
                frame_data_scaled = (frame_data - emg_means) / emg_stds
                # Extract features from the EMG data
                frame_features = extract_emg_features(frame_data_scaled)
                features.extend(frame_features)
            feature_vector = np.array(features).reshape(1, -1)

            # Scale the features
            feature_vector_scaled = feature_scaler.transform(feature_vector)

            # Make predictions using the neural network model
            predictions_prob = model.predict(feature_vector_scaled)
            predictions_binary = (predictions_prob > 0.5).astype(int)[0]

            # Smooth predictions
            smoothed_predictions = smooth_predictions(predictions_binary, smoothing_buffer, SMOOTHING_WINDOW_SIZE)

            # Convert predictions to colored dots for display
            thumb_dot = f"{COLOR_GREEN}●{COLOR_RESET}" if smoothed_predictions[0] else f"{COLOR_GRAY}●{COLOR_RESET}"
            index_middle_dot = f"{COLOR_GREEN}●{COLOR_RESET}" if smoothed_predictions[1] else f"{COLOR_GRAY}●{COLOR_RESET}"
            ring_pinky_dot = f"{COLOR_GREEN}●{COLOR_RESET}" if smoothed_predictions[2] else f"{COLOR_GRAY}●{COLOR_RESET}"

            # Display the predictions
            clear_console()
            print("\nPredictions (Smoothed):")
            print(f"Thumb:          {thumb_dot}")
            print(f"Index/Middle:   {index_middle_dot}")
            print(f"Ring/Pinky:     {ring_pinky_dot}")

            # Optional small delay
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stop_event.set()
        emg_thread.join()

if __name__ == "__main__":
    main()
