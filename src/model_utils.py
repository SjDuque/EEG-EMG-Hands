# model_utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import register_keras_serializable

# model_utils.py
from scipy.signal import resample

def resample_dataframe_signal(df, desired_rate, time_column='timestamp'):
    # Ensure the time column is sorted
    df = df.sort_values(by=time_column).reset_index(drop=True)
    
    # Remove rows where all data columns are NaN
    data_columns = df.columns.drop(time_column)
    df = df.dropna(subset=data_columns, how='all').reset_index(drop=True)
    
    # Original sampling rate
    time_values = df[time_column].values
    start_time = time_values[0]
    end_time = time_values[-1]
    original_rate = 1.0 / np.mean(np.diff(time_values))
    
    num_samples = int((end_time - start_time) * desired_rate)
    new_time_values = np.linspace(start_time, end_time, num=num_samples)
    
    resampled_data = {time_column: new_time_values}
    
    for col in data_columns:
        if df[col].isna().all():
            resampled_data[col] = np.full_like(new_time_values, np.nan, dtype=np.float64)
            continue
        # Interpolate using resample, ignoring NaNs
        signal = df[col].interpolate().fillna(method='bfill').fillna(method='ffill').values
        resampled_signal = resample(signal, num_samples)
        resampled_data[col] = resampled_signal
    
    resampled_df = pd.DataFrame(resampled_data)
    return resampled_df


def resample_dataframe(df, desired_rate, time_column='timestamp', method='linear'):
    """
    Resample the DataFrame to the desired sampling rate without creating data where it doesn't exist.
    
    Parameters:
    - df: pandas DataFrame containing the data to resample.
    - desired_rate: Desired sampling rate in Hz.
    - time_column: Name of the time column in df.
    - method: Interpolation method ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.)
    
    Returns:
    - resampled_df: DataFrame resampled to the desired rate.
    """
    # Ensure the time column is sorted
    df = df.sort_values(by=time_column).reset_index(drop=True)
    
    # Remove rows where all data columns are NaN
    data_columns = df.columns.drop(time_column)
    df = df.dropna(subset=data_columns, how='all').reset_index(drop=True)
    
    # Get the time values where data is available
    time_values = df[time_column].values
    
    # Determine the new time values
    start_time = time_values[0]
    end_time = time_values[-1]
    
    # Generate new time values at desired_rate
    sampling_interval = 1.0 / desired_rate
    new_time_values = np.arange(start_time, end_time, sampling_interval)
    
    # Interpolate data columns onto new time grid
    from scipy.interpolate import interp1d
    
    resampled_data = {time_column: new_time_values}
    
    for col in data_columns:
        # Mask NaN values
        valid_mask = ~np.isnan(df[col].values)
        if np.sum(valid_mask) < 2:
            # Not enough data to interpolate, fill with NaN
            resampled_data[col] = np.full_like(new_time_values, np.nan, dtype=np.float64)
            continue
        
        # Create interpolation function only with valid data
        interp_func = interp1d(
            time_values[valid_mask],
            df[col].values[valid_mask],
            kind=method,
            bounds_error=False,
            fill_value=np.nan  # Keep NaN where data is missing
        )
        # Interpolate onto new time values
        resampled_data[col] = interp_func(new_time_values)
    
    resampled_df = pd.DataFrame(resampled_data)
    
    return resampled_df


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

def build_cnn_model(input_shape, output_dim, class_weights=None):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense

    inputs = Input(shape=input_shape)
    x = Conv1D(filters=512, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x) 
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    # x = Dense(16, activation='relu')(x)
    # x = Dropout(0.5)(x)
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
