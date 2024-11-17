import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Load your EMG data (assuming it's a CSV file)
emg_data = pd.read_csv("data/EMG Hand Data 20241030_223524/emg.csv")

# Step 1: Normalize EMG data
def normalize_emg(data):
    # Z-score normalization
    return (data - data.mean()) / data.std()

normalized_emg = emg_data.iloc[:, 1:].apply(normalize_emg)

# Step 2: Rectify and Low-Pass Filter
def rectify_and_filter(emg_signal, cutoff=4, fs=250):
    # Rectify
    rectified_signal = np.abs(emg_signal)
    # Low-pass filter
    b, a = butter(2, cutoff / (0.5 * fs), btype='low')
    filtered_signal = filtfilt(b, a, rectified_signal)
    return filtered_signal

filtered_emg = normalized_emg.apply(lambda x: rectify_and_filter(x))

# Step 3: Calculate Muscle Activation Values with Recursive Filtering
def muscle_activation_model(emg_signal, alpha=0.5, beta1=0.3, beta2=0.1, delay=1):
    # Initialize muscle activation array
    activation = np.zeros(len(emg_signal))
    for t in range(delay, len(emg_signal)):
        activation[t] = (alpha * emg_signal[t - delay]
                         - beta1 * activation[t - 1]
                         - beta2 * activation[t - 2] if t - 2 >= 0 else 0)
    return activation

# Apply muscle activation model to each channel
activation_values = filtered_emg.apply(muscle_activation_model)

# Result: muscle activation values are now in `activation_values`
print(activation_values)