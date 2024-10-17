# data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import logging
import os
import json
import datetime

# Configuration Parameters
SAMPLING_RATE = 250
SELECTED_CHANNELS = [1, 2, 3, 4]
AUGMENT_FACTOR = 2
FRAME_SIZE_MS = 100
CLOSED_THRESHOLDS = {
    'pinky': 0.5,
    'ring': 0.5,
    'middle': 0.5,
    'index': 0.5,
    'thumb': 0.5
}
JOINT_ANGLE_THRESHOLDS = {
    'thumb': (132.5, 157.5),
    'index': (25, 165),
    'middle': (15, 167.5),
    'ring': (15, 167.5),
    'pinky': (15, 167.5)
}
AUGMENTATIONS = {
    "magnitude_warp": True,
    "time_warp": True,
    "permutation": True,
    "add_gaussian_noise": True,
    "scale_time_series": True
}
AGGREGATE_FINGERS = [['pinky', 'ring'], ['middle', 'index'], 'thumb']

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load and preprocess data
def load_and_preprocess_data(hand_file: str, emg_file: str, sampling_rate: int = None, selected_channels: list = None) -> tuple:
    sampling_rate = sampling_rate or SAMPLING_RATE
    selected_channels = selected_channels or SELECTED_CHANNELS

    try:
        hand_df = pd.read_csv(hand_file)
        emg_df = pd.read_csv(emg_file)
        logging.info("Files loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        print(f"File not found: {e}")
        return 

    hand_df = hand_df.sort_values('timestamp').reset_index(drop=True)
    emg_df = emg_df.sort_values('timestamp').reset_index(drop=True)

    hand_start_index = int(sampling_rate * 1 / 15)
    hand_df = hand_df.iloc[hand_start_index:].reset_index(drop=True)

    for joint, (low, high) in JOINT_ANGLE_THRESHOLDS.items():
        angle_column = f"{joint}_angle"
        percent_column = f"{joint}_percent"
        hand_df[percent_column] = hand_df[angle_column].apply(lambda x: (x - low) / (high - low))

    emg_channels = [f'emg_channel_{i}' for i in selected_channels]
    emg_df_selected = emg_df[emg_channels].copy()
    emg_timestamps = emg_df['timestamp'].reset_index(drop=True)

    scaler_emg = MinMaxScaler()
    emg_df_scaled = pd.DataFrame(scaler_emg.fit_transform(emg_df_selected), columns=emg_channels)
    emg_df_final = pd.concat([emg_timestamps, emg_df_scaled], axis=1)

    return hand_df, emg_df_final, scaler_emg

# Data augmentation functions
def magnitude_warp(x: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
    orig_steps = np.arange(x.shape[0])
    warp_steps = np.linspace(0, x.shape[0]-1, knot)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[1], knot))
    warped = np.zeros_like(x)
    
    for i in range(x.shape[1]):
        cs = CubicSpline(warp_steps, random_warps[i])
        factor = cs(orig_steps)
        warped[:, i] = x[:, i] * factor
    return warped

def time_warp(x: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
    orig_steps = np.arange(x.shape[0])
    warp_steps = np.linspace(0, x.shape[0]-1, knot)
    random_warps = np.random.normal(loc=0.0, scale=sigma, size=(x.shape[1], knot))
    warped = np.zeros_like(x)
    
    for i in range(x.shape[1]):
        cs = CubicSpline(warp_steps, random_warps[i])
        warp = cs(orig_steps)
        steps = orig_steps + warp
        steps = np.clip(steps, 0, x.shape[0]-1)
        warped[:, i] = np.interp(orig_steps, steps, x[:, i])
    return warped

def permutation(x: np.ndarray, max_segments: int = 5) -> np.ndarray:
    orig_steps = x.shape[0]
    num_segs = np.random.randint(1, max_segments)
    seg_size = orig_steps // num_segs
    segments = []
    for i in range(num_segs):
        start = i * seg_size
        end = (i + 1) * seg_size if i < num_segs - 1 else orig_steps
        segments.append(x[start:end, :])
    np.random.shuffle(segments)
    return np.vstack(segments)

def add_gaussian_noise(x: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0, noise_level, x.shape)
    return x + noise

def scale_time_series(x: np.ndarray, scale_low: float = 0.9, scale_high: float = 1.1) -> np.ndarray:
    scale = np.random.uniform(scale_low, scale_high, (1, x.shape[1]))
    return x * scale

# Augment data function with control over techniques
def augment_data(X: np.ndarray, y: np.ndarray, augment_factor: int = None, augmentations: dict = None) -> tuple:
    augment_factor = augment_factor or AUGMENT_FACTOR
    augmentations = augmentations or AUGMENTATIONS

    X_augmented, y_augmented = [], []
    
    for i in range(len(X)):
        original_shape = X[i].shape
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        
        for _ in range(augment_factor):
            augmented = X[i].copy()

            if augmentations["magnitude_warp"] and np.random.rand() > 0.5:
                augmented = magnitude_warp(augmented)
            if augmentations["time_warp"] and np.random.rand() > 0.5:
                augmented = time_warp(augmented)
            if augmentations["permutation"] and np.random.rand() > 0.5:
                augmented = permutation(augmented)
            if augmentations["add_gaussian_noise"] and np.random.rand() > 0.5:
                augmented = add_gaussian_noise(augmented)
            if augmentations["scale_time_series"] and np.random.rand() > 0.5:
                augmented = scale_time_series(augmented)

            # Ensure augmented data has the original shape
            if augmented.shape == original_shape:
                X_augmented.append(augmented)
                y_augmented.append(y[i])
            else:
                logging.warning(f"Augmented data shape {augmented.shape} does not match original shape {original_shape}")

    return np.array(X_augmented), np.array(y_augmented)

# Create aggregated binary frames
def create_aggregated_binary_frames(
    hand_df: pd.DataFrame, 
    emg_df: pd.DataFrame, 
    emg_columns: list = None, 
    frame_size_ms: int = None, 
    sampling_rate: int = None, 
    closed_thresholds: dict = None, 
    aggregate_method: list = None  
) -> tuple:
    emg_columns = emg_columns or [f'emg_channel_{i}' for i in SELECTED_CHANNELS]
    frame_size_ms = frame_size_ms or FRAME_SIZE_MS
    sampling_rate = sampling_rate or SAMPLING_RATE
    closed_thresholds = closed_thresholds or CLOSED_THRESHOLDS
    aggregate_method = aggregate_method or AGGREGATE_FINGERS

    frame_size_samples = int((frame_size_ms / 1000) * sampling_rate)
    X_frames, y_frames = [], []

    emg_df = emg_df.set_index('timestamp')

    for idx, row in hand_df.iterrows():
        timestamp = row['timestamp']
        emg_indices = emg_df.index[emg_df.index <= timestamp]
        if not emg_indices.empty:
            emg_idx = emg_indices[-1]
        else:
            continue

        emg_position = emg_df.index.get_loc(emg_idx)

        if emg_position >= frame_size_samples:
            frame_data = emg_df.iloc[emg_position - frame_size_samples:emg_position][emg_columns].values
            X_frames.append(frame_data)

            binary_labels = []
            for group in aggregate_method:
                if isinstance(group, list):
                    group_min = row[[f"{finger}_percent" for finger in group]].min()
                    group_label = int(group_min < min(closed_thresholds[finger] for finger in group))
                else:
                    group_label = int(row[f"{group}_percent"] < closed_thresholds[group])
                binary_labels.append(group_label)

            y_frames.append(binary_labels)

    logging.info(f"Number of frames created: {len(X_frames)}")
    return np.array(X_frames), np.array(y_frames).astype(np.float32)

# Function to save the model and preprocessing information
def save_model_and_preprocessing(
    model, scaler_emg, best_thresholds, frame_size_ms: int = None, sampling_rate: int = None
):
    frame_size_ms = frame_size_ms or FRAME_SIZE_MS
    sampling_rate = sampling_rate or SAMPLING_RATE

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"EMG_Model_{now}"
    os.makedirs(folder_name, exist_ok=True)

    model_file = os.path.join(folder_name, 'emg_lstm_model.keras')
    model.save(model_file)
    print(f"Model saved to {model_file}")

    best_thresholds_float = [float(threshold) for threshold in best_thresholds]

    preprocessing_info = {
        'scaler_emg_min': scaler_emg.min_.tolist(),
        'scaler_emg_scale': scaler_emg.scale_.tolist(),
        'frame_size_ms': frame_size_ms,
        'sampling_rate': sampling_rate,
        'best_thresholds': best_thresholds_float,
        'joint_angle_thresholds': JOINT_ANGLE_THRESHOLDS,
        'augmentations': AUGMENTATIONS
    }

    preprocessing_file = os.path.join(folder_name, 'preprocessing_info.json')
    with open(preprocessing_file, 'w') as f:
        json.dump(preprocessing_info, f)
    print(f"Preprocessing information saved to {preprocessing_file}")
