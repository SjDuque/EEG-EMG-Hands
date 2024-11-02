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

# Existing functions from model.py
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
    x = Conv1D(filters=48, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x) 
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    # x = MaxPooling1D(pool_size=2)(x) 
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    
    x = Flatten()(x)
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
