# model_utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import resample, butter, filtfilt
from scipy.interpolate import interp1d
from itertools import product
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss, BinaryCrossentropy, BinaryFocalCrossentropy
from tensorflow.keras.utils import register_keras_serializable, Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense, GaussianNoise, Conv2D, MaxPooling2D
import logging
from sklearn.metrics import classification_report, confusion_matrix, r2_score

@register_keras_serializable()
class WeightedBinaryCrossentropy(Loss):
    def __init__(self, class_weights, name='weighted_binary_crossentropy'):
        super().__init__(name=name)
        # Assuming class_weights is a dict where keys are class indices
        # and values are dicts mapping class labels to weights
        self.class_weights_neg = tf.constant([class_weights[i][0] for i in range(len(class_weights))], dtype=tf.float32)
        self.class_weights_pos = tf.constant([class_weights[i][1] for i in range(len(class_weights))], dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        # Compute binary cross-entropy
        bce = K.binary_crossentropy(y_true, y_pred)
        
        # Compute weights: y_true * weight_pos + (1 - y_true) * weight_neg
        weight = y_true * self.class_weights_pos + (1 - y_true) * self.class_weights_neg
        
        # Apply weights to the loss
        weighted_bce = bce * weight
        
        # Return the mean loss over all classes and samples
        return K.mean(weighted_bce)

@register_keras_serializable()
class WeightedMSE(Loss):
    def __init__(self, output_weights, name='weighted_mse'):
        super().__init__(name=name)
        self.output_weights = tf.constant(output_weights, dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        mse = K.square(y_true - y_pred)
        weighted_mse = mse * self.output_weights
        return K.mean(weighted_mse)

def low_pass_filter(data, cutoff=4, fs=250, order=2):
    """
    Applies a low-pass Butterworth filter to the data.

    Parameters:
    - data: 1D numpy array of the signal to filter.
    - cutoff: Cutoff frequency in Hz.
    - fs: Sampling frequency in Hz.
    - order: Order of the Butterworth filter.

    Returns:
    - Filtered data as a 1D numpy array.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def estimate_sampling_frequency(timestamps):
    """
    Estimates the sampling frequency based on timestamps.

    Parameters:
    - timestamps: 1D numpy array of timestamps.

    Returns:
    - Estimated sampling frequency as an integer.
    """
    sampling_intervals = np.diff(timestamps)
    return int(1 / np.mean(sampling_intervals))

def resample_dataframe_signal(df, desired_rate, time_column='timestamp'):
    """
    Resamples the DataFrame signals to a desired sampling rate.

    Parameters:
    - df: pandas DataFrame with a time column.
    - desired_rate: Target sampling rate in Hz.
    - time_column: Name of the time column.

    Returns:
    - Resampled pandas DataFrame.
    """
    # Ensure the time column is sorted
    df = df.sort_values(by=time_column).reset_index(drop=True)
    
    # Remove rows where all data columns are NaN
    data_columns = df.columns.drop(time_column)
    df = df.dropna(subset=data_columns, how='all').reset_index(drop=True)
    
    time_values = df[time_column].values
    start_time, end_time = time_values[0], time_values[-1]
    num_samples = int((end_time - start_time) * desired_rate)
    new_time_values = np.linspace(start_time, end_time, num=num_samples)
    
    resampled_data = {time_column: new_time_values}
    
    for col in data_columns:
        if df[col].isna().all():
            resampled_data[col] = np.full_like(new_time_values, np.nan, dtype=np.float64)
        else:
            signal = df[col].interpolate(method='linear').values
            resampled_signal = resample(signal, num_samples)
            resampled_data[col] = resampled_signal
    
    return pd.DataFrame(resampled_data)

def resample_dataframe(df, desired_rate, time_column='timestamp', method='linear'):
    """
    Resample the DataFrame to the desired sampling rate.

    Parameters:
    - df: pandas DataFrame with a time column.
    - desired_rate: Target sampling rate in Hz.
    - time_column: Name of the time column.
    - method: Interpolation method.

    Returns:
    - Resampled DataFrame.
    """
    # Sort and clean the DataFrame
    df = df.sort_values(by=time_column).reset_index(drop=True)
    data_columns = df.columns.drop(time_column)
    df = df.dropna(subset=data_columns, how='all').reset_index(drop=True)
    
    time_values = df[time_column].values
    start_time, end_time = time_values[0], time_values[-1]
    sampling_interval = 1.0 / desired_rate
    new_time_values = np.arange(start_time, end_time, sampling_interval)
    
    resampled_data = {time_column: new_time_values}
    
    for col in data_columns:
        valid = ~np.isnan(df[col].values)
        if np.sum(valid) < 2:
            resampled_data[col] = np.full_like(new_time_values, np.nan, dtype=np.float64)
            continue
        interp_func = interp1d(
            time_values[valid],
            df[col].values[valid],
            kind=method,
            bounds_error=False,
            fill_value=np.nan
        )
        resampled_data[col] = interp_func(new_time_values)
    
    return pd.DataFrame(resampled_data)

def smooth_emg_channel(emg_channel, beta=0.9, emg_fs=250):
    """
    Smooths a single EMG channel using exponential smoothing

    Parameters:
    - emg_channel: 1D numpy array of EMG data.
    - beta: Smoothing factor between 0 and 1.
    - emg_fs: Sampling frequency in Hz.

    Returns:
    - smoothed_emg: 1D numpy array of smoothed EMG data.
    """
    smoothed_emg = np.zeros_like(emg_channel)
    delay = int(0.05 * emg_fs)  # Delay in samples
    
    # Apply exponential smoothing
    for t in range(delay, len(emg_channel)):
        smoothed_emg[t] = beta * emg_channel[t-delay] + (1 - beta) * smoothed_emg[t - 1]
        
    smoothed_emg[:delay] = smoothed_emg[delay]

    return smoothed_emg

def smooth_emg(emg_data, beta=0.9, emg_fs=250):
    """
    Smooths all EMG channels in a DataFrame with the same beta and delay_ms.

    Parameters:
    - emg_data: pandas DataFrame containing EMG channels.
    - beta: Smoothing factor.
    - delay_ms: Delay in milliseconds.
    - emg_fs: Sampling frequency in Hz.

    Returns:
    - smoothed_emg_data: pandas DataFrame with smoothed EMG channels.
    """
    smoothed_emg_data = emg_data.copy()
    for channel in emg_data.columns:
        smoothed_emg_data[channel] = smooth_emg_channel(
            emg_data[channel].values, beta, emg_fs
        )
    return smoothed_emg_data

def create_model(input_shape, output_shape, gaussian_noise=0.1):
    """
    Builds and compiles a Sequential Keras model with Gaussian Noise and several Dense layers.

    Parameters:
    - input_shape: Tuple representing the shape of the input data.
    - output_shape: Number of output neurons.
    - gaussian_noise: Standard deviation for GaussianNoise layer.

    Returns:
    - Compiled Keras model.
    """
    model = Sequential([
        GaussianNoise(gaussian_noise, input_shape=input_shape),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(output_shape, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['r2_score'])
    return model

def build_cnn_model(input_shape, output_dim, class_weights=None, noise_stddev=0.1):
    """
    Builds a Convolutional Neural Network (CNN) model for classification.

    Parameters:
    - input_shape: Tuple representing the shape of the input data.
    - output_dim: Number of output neurons.
    - class_weights: Dictionary containing class weights for each output.

    Returns:
    - Compiled Keras CNN model.
    """
    inputs = Input(shape=input_shape)
    
    # x = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(inputs)
    # x = MaxPooling1D(pool_size=2)(x) 
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    
    # x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    # x = MaxPooling1D(pool_size=2)(x) 
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    
    # x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    # x = MaxPooling1D(pool_size=2)(x) 
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x) 
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    # x = MaxPooling1D(pool_size=2)(x) 
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    outputs = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    if class_weights:
        loss_fn = WeightedBinaryCrossentropy(class_weights)
    else:
        loss_fn = BinaryCrossentropy()
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['binary_accuracy', 'r2_score'])
    return model

def build_cnn_model_stacked_channels(input_shape, output_dim, class_weights=None):
    """
    Builds a CNN model with stacked channels for classification.

    Parameters:
    - input_shape: Tuple representing the shape of the input data.
    - output_dim: Number of output neurons.
    - class_weights: Dictionary containing class weights for each output.

    Returns:
    - Compiled Keras CNN model with stacked channels.
    """
    inputs = Input(shape=input_shape)
    
    # Reshape input to add a channel dimension for 2D convolutions
    x = tf.keras.layers.Reshape((*input_shape, 1))(inputs)
    
    x = Conv2D(filters=16, kernel_size=(3, input_shape[1]), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(filters=32, kernel_size=(3, input_shape[1]), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(filters=64, kernel_size=(3, input_shape[1]), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    if class_weights:
        loss_fn = WeightedBinaryCrossentropy(class_weights)
    else:
        loss_fn = BinaryCrossentropy()
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['binary_accuracy', 'r2_score'])
    return model

def plot_learning_curves(history):
    """
    Plots the learning curves for training and validation metrics.

    Parameters:
    - history: Keras History object returned by model.fit().
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('binary_accuracy', history.history.get('accuracy')), label='Train Binary Accuracy')
    plt.plot(history.history.get('val_binary_accuracy', history.history.get('val_accuracy')), label='Validation Binary Accuracy')
    plt.title('Model Binary Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def create_labels(fingers_angles_df, finger_groups, thresholds):
    """
    Creates binary labels based on finger angles and predefined thresholds.

    Parameters:
    - fingers_angles_df: DataFrame containing finger angles.
    - finger_groups: Dictionary mapping group names to list of fingers.
    - thresholds: Dictionary mapping finger names to threshold values.

    Returns:
    - labels: 2D numpy array of binary labels.
    - group_names: List of group names.
    """
    labels = np.zeros((len(fingers_angles_df), len(finger_groups)), dtype=int)
    group_names = list(finger_groups.keys())
    for i, (group_name, fingers) in enumerate(finger_groups.items()):
        # Ensure thresholds use uppercase keys
        threshold_values = [thresholds[f.upper()] for f in fingers]
        group_condition = np.any(
            fingers_angles_df[fingers].lt(threshold_values, axis=1),
            axis=1
        )
        labels[:, i] = group_condition.astype(int)
    return labels, group_names

def find_nearest_previous_timestamp(emg_timestamps, target_timestamp):
    """
    Finds the index of the nearest previous timestamp in EMG data for a given target timestamp.

    Parameters:
    - emg_timestamps: 1D numpy array of EMG timestamps.
    - target_timestamp: Target timestamp to align.

    Returns:
    - Index of the nearest previous timestamp or None if not found.
    """
    idx = np.searchsorted(emg_timestamps, target_timestamp, side='right') - 1
    return idx if idx >= 0 else None

def evaluate_model(model, X, y, group_names, label='Test Set'):
    """
    Evaluate the model and print classification metrics, including R² score.

    Parameters:
    - model: Trained Keras model.
    - X: Features for evaluation.
    - y: True labels for evaluation.
    - group_names: Names of the finger groups.
    - label: Descriptor for the evaluation set.
    """
    loss, accuracy, _ = model.evaluate(X, y, verbose=0)
    logging.info(f"{label} Accuracy: {accuracy * 100:.2f}%")
    logging.info(f"{label} Loss: {loss:.4f}")

    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print(f"\n{label} Classification Report:")
    print(classification_report(
        y, y_pred,
        target_names=group_names,
        zero_division=0
    ))

    for i, finger_group in enumerate(group_names):
        print(f"\nConfusion Matrix for {finger_group} ({label}):")
        cm = confusion_matrix(y[:, i], y_pred[:, i])
        print(cm)

    # Calculate R² scores
    r2_scores = r2_score(y, y_pred_prob, multioutput='raw_values')
    for i, finger_group in enumerate(group_names):
        print(f"R² score for {finger_group} ({label}): {r2_scores[i]:.4f}")

class AugmentedDataGenerator(Sequence):
    """
    Keras Sequence generator for data augmentation.
    """
    def __init__(self, X, y, batch_size=32, shuffle=True, 
                 add_noise=False, noise_stddev=0.05,
                 apply_time_warp=False, max_warp=0.2,
                 apply_magnitude_scaling=False, scale_range=(0.8, 1.2),):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.add_noise = add_noise
        self.noise_stddev = noise_stddev
        self.apply_time_warp = apply_time_warp
        self.max_warp = max_warp
        self.apply_magnitude_scaling = apply_magnitude_scaling
        self.scale_range = scale_range
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices]
        
        if self.add_noise or self.apply_time_warp or self.apply_magnitude_scaling:
            X_batch = self._apply_augmentations(X_batch)
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _apply_augmentations(self, X_batch):
        for i in range(X_batch.shape[0]):
            if self.add_noise:
                X_batch[i] += np.random.normal(0, self.noise_stddev, X_batch[i].shape)
            
            if self.apply_time_warp:
                X_batch[i] = self._time_warp(X_batch[i])
            
            if self.apply_magnitude_scaling:
                X_batch[i] *= np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        return X_batch
    
    def _time_warp(self, sequence):
        num_frames, num_channels = sequence.shape
        warp_factor = np.random.uniform(1 - self.max_warp, 1 + self.max_warp)
        warped_num_frames = max(int(num_frames * warp_factor), 2)
        warped_sequence = resample(sequence, warped_num_frames, axis=0)
        warped_sequence = resample(warped_sequence, num_frames, axis=0)
        return warped_sequence

def print_label_distribution(y, group_names, label='Overall'):
    """
    Prints the distribution of labels for each finger group.

    Parameters:
    - y: 2D numpy array of binary labels.
    - group_names: List of group names corresponding to label columns.
    - label: Descriptor for the dataset being printed (e.g., 'Training Set').
    """
    total_samples = y.shape[0]
    label_sums = np.sum(y, axis=0)
    print(f"\nLabel distribution ({label}):")
    for i, finger_group in enumerate(group_names):
        count = label_sums[i]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"{finger_group}: {count}/{total_samples} ({percentage:.2f}%)")
