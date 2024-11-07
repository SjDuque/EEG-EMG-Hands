# model_utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import resample, butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import register_keras_serializable, Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense, GaussianNoise
import logging

class AugmentedDataGenerator(Sequence):
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
        
        if self.add_noise or self.apply_time_warp or self.apply_magnitude_scaling or self.apply_bandpass_filter:
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

def resample_dataframe_signal(df, desired_rate, time_column='timestamp'):
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

def find_nearest_previous_timestamp(emg_timestamps, target_timestamp):
    idx = np.searchsorted(emg_timestamps, target_timestamp, side='right') - 1
    return idx if idx >= 0 else None

def create_labels(fingers_angles_df, finger_groups, thresholds):
    labels = np.zeros((len(fingers_angles_df), len(finger_groups)), dtype=int)
    group_names = list(finger_groups.keys())
    for i, (group_name, fingers) in enumerate(finger_groups.items()):
        # Ensure thresholds use lowercase keys
        threshold_values = [thresholds[f.upper()] for f in fingers]
        group_condition = np.any(
            fingers_angles_df[fingers].lt(threshold_values, axis=1),
            axis=1
        )
        labels[:, i] = group_condition.astype(int)
    return labels, group_names

def print_label_distribution(y, group_names, label='Overall'):
    total_samples = y.shape[0]
    label_sums = np.sum(y, axis=0)
    print(f"\nLabel distribution ({label}):")
    for i, finger_group in enumerate(group_names):
        count = label_sums[i]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"{finger_group}: {count}/{total_samples} ({percentage:.2f}%)")

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

def build_cnn_model(input_shape, output_dim, class_weights=None, noise_stddev=0.1):
    inputs = Input(shape=input_shape)
    
    # Add Gaussian Noise Layer
    # x = GaussianNoise(noise_stddev)(inputs)  # Adjust noise_stddev as needed
    
    x = Conv1D(filters=512, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x) 
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # x = Conv1D(filters=512, kernel_size=3, activation='relu', padding='same')(inputs)
    # x = MaxPooling1D(pool_size=2)(x) 
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    
    # x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(inputs)
    # x = MaxPooling1D(pool_size=2)(x) 
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    # x = Dense(512, activation='relu')(x)
    outputs = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    if class_weights:
        loss_fn = WeightedBinaryCrossentropy(class_weights)
    else:
        loss_fn = 'binary_crossentropy'
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['binary_accuracy'])
    return model

def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['binary_accuracy'], label='Train Binary Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Binary Accuracy')
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
