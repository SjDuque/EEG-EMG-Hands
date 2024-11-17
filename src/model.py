# main_script.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Conv1D, MaxPooling1D, Flatten, LSTM, Add, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os

from model_utils import (
    find_nearest_previous_timestamp,
    create_labels,
    print_label_distribution,
    build_cnn_model,
    build_cnn_model_stacked_channels,
    plot_learning_curves,
    resample_dataframe,
    resample_dataframe_signal,
    AugmentedDataGenerator,
    low_pass_filter,
    smooth_emg
)

# Configuration Parameters
SELECTED_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]
FINGER_GROUPS = {
    'Thumb': ['THUMB'], 
    'Index': ['INDEX'],
    'Middle': ['MIDDLE'],
    'Ring': ['RING'],
    'Pinky': ['PINKY']
    # Add more finger groups as needed
    # 'Index_Middle': ['INDEX', 'MIDDLE'],
    # 'Ring_Pinky': ['RING', 'PINKY'],
}

JOINT_ANGLE_THRESHOLDS = {
    'THUMB':  (132.5 + 157.5) / 2,
    'INDEX':  (60 + 160) / 2,
    'MIDDLE': (40 + 167.5) / 2,
    'RING':   (30 + 167.5) / 2,
    'PINKY':  (30 + 167.5) / 2
}

# Sampling Rates: Resample EMG and Finger data
EMG_SAMPLING_RATE = 250  # Hz
FINGER_SAMPLING_RATE = 100  # Hz

# New Configuration Parameters for Frame-Based Processing
NUM_FRAMES = EMG_SAMPLING_RATE//2  # Number of EMG frames to include before the timestamp
FRAME_STEP = 1   # Step size between frames

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def save_model_and_preprocessing(model, scaler, emg_means, emg_stds, model_dir):
    """
    Saves the trained model and the EMG scaler to the specified directory.

    Parameters:
    - model: Trained Keras model.
    - scaler: Fitted StandardScaler object.
    - emg_means: Pandas Series containing means of EMG channels.
    - emg_stds: Pandas Series containing standard deviations of EMG channels.
    - model_dir: Directory path to save the model and scaler.
    """
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'model.keras'))
    # Save the EMG scaler
    joblib.dump(scaler, os.path.join(model_dir, 'emg_scaler.joblib'))
    # Save Configuration Parameters
    config = {
        'NUM_FRAMES': NUM_FRAMES,
        'FRAME_STEP': FRAME_STEP,
        'FINGER_GROUPS': FINGER_GROUPS,
        'SELECTED_CHANNELS': SELECTED_CHANNELS,
        'JOINT_ANGLE_THRESHOLDS': JOINT_ANGLE_THRESHOLDS,
        'EMG_MEANS': emg_means,
        'EMG_STDS': emg_stds,
    }
    joblib.dump(config, os.path.join(model_dir, 'config.joblib'))
    
    logging.info(f"Model and preprocessing saved to {model_dir}.")

def process_sessions_frame_based(session_dirs, num_frames, emg_means=None, emg_stds=None, use_existing_scaler=False):
    """
    Process EMG and finger angle data from session directories using frame-based approach.

    Parameters:
    - session_dirs: List of session directory paths.
    - num_frames: Number of frames to include before the timestamp.
    - emg_means: Precomputed means for EMG channels (required if use_existing_scaler is True).
    - emg_stds: Precomputed standard deviations for EMG channels (required if use_existing_scaler is True).
    - use_existing_scaler: Boolean indicating whether to use existing normalization parameters.

    Returns:
    - If use_existing_scaler is False:
        - X: NumPy array of extracted frame sequences.
        - y: NumPy array of binary labels.
        - skipped: Number of samples skipped due to missing data.
        - emg_means: Means of EMG channels.
        - emg_stds: Standard deviations of EMG channels.
    - If use_existing_scaler is True:
        - X: NumPy array of extracted frame sequences.
        - y: NumPy array of binary labels.
        - skipped: Number of samples skipped due to missing data.
    """
    all_sequences = []
    all_labels = []
    skipped = 0
    emg_channels = [f'emg_channel_{ch}' for ch in SELECTED_CHANNELS]

    for session_dir in session_dirs:
        emg_file = os.path.join(session_dir, 'emg.csv')
        fingers_file = os.path.join(session_dir, 'finger_angles.csv')

        try:
            emg_df = pd.read_csv(emg_file)
            fingers_df = pd.read_csv(fingers_file)
            logging.info(f"Data files loaded successfully for session {session_dir}.")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            continue
        
        # Resample EMG data
        emg_df = resample_dataframe_signal(emg_df, desired_rate=EMG_SAMPLING_RATE)
        logging.info(f"EMG data resampled for session {session_dir}.")
        # Resample Finger data
        fingers_df = resample_dataframe(fingers_df, desired_rate=FINGER_SAMPLING_RATE, method='linear')
        logging.info(f"Finger data resampled for session {session_dir}.")
        
        expected_emg_columns = ['timestamp'] + emg_channels
        if not all(col in emg_df.columns for col in expected_emg_columns):
            logging.error(f"EMG data file in {session_dir} is missing required columns.")
            continue

        expected_finger_columns = ['timestamp'] + list(JOINT_ANGLE_THRESHOLDS.keys())
        if not all(col in fingers_df.columns for col in expected_finger_columns):
            logging.error(f"Finger data file in {session_dir} is missing required columns.")
            continue

        # Drop rows with missing data
        fingers_df.dropna(subset=expected_finger_columns, inplace=True)
        emg_df.dropna(subset=expected_emg_columns, inplace=True)
        
        # Rectify negative EMG values using abs
        emg_df[emg_channels] = emg_df[emg_channels].abs()

        if not use_existing_scaler:
            current_emg_means = emg_df[emg_channels].mean()
            current_emg_stds = emg_df[emg_channels].std()
            emg_df[emg_channels] = (emg_df[emg_channels] - current_emg_means) / current_emg_stds
            logging.info(f"EMG data normalized for session {session_dir}.")
        else:
            if emg_means is None or emg_stds is None:
                raise ValueError("emg_means and emg_stds must be provided when use_existing_scaler is True.")
            emg_df[emg_channels] = (emg_df[emg_channels] - emg_means) / emg_stds
            logging.info(f"EMG data normalized using existing scaler for session {session_dir}.")

        # Apply low-pass filter to EMG channels
        for channel in emg_channels:
            emg_df[channel] = low_pass_filter(emg_df[channel].values, cutoff=4, fs=EMG_SAMPLING_RATE, order=2)
        logging.info(f"Low-pass filter applied to EMG channels for session {session_dir}.")

        # Apply smoothing to EMG channels
        emg_df[emg_channels] = smooth_emg(emg_df[emg_channels], beta=0.95, emg_fs=EMG_SAMPLING_RATE)
        logging.info(f"Smoothing applied to EMG channels for session {session_dir}.")

        emg_data = emg_df[emg_channels].values
        emg_timestamps = emg_df['timestamp'].values
        logging.info(f"EMG data extracted for session {session_dir}.")

        fingers_timestamps = fingers_df['timestamp'].values
        fingers_angles = fingers_df[expected_finger_columns[1:]]
        logging.info(f"Finger data extracted for session {session_dir}.")

        for idx, finger_timestamp in enumerate(fingers_timestamps):
            nearest_emg_idx = find_nearest_previous_timestamp(emg_timestamps, finger_timestamp)
            if nearest_emg_idx is None:
                logging.warning(f"No EMG data before finger timestamp {finger_timestamp} in session {session_dir}. Skipping.")
                skipped += 1
                continue

            # Calculate start index for the sequence
            start_idx = nearest_emg_idx - num_frames + 1
            end_idx = nearest_emg_idx + 1  # Exclusive

            if start_idx < 0:
                logging.warning(f"Not enough EMG data before timestamp {finger_timestamp} in session {session_dir}. Skipping.")
                skipped += 1
                continue

            frame_sequence = emg_data[start_idx:end_idx, :]  # Shape: (num_frames, num_channels)
            all_sequences.append(frame_sequence)
            all_labels.append(fingers_angles.iloc[idx])

    X = np.array(all_sequences)  # Shape: (num_samples, num_frames, num_channels)
    logging.info(f"Frame sequence extraction completed. Shape: {X.shape}")

    fingers_angles_df = pd.DataFrame(all_labels)
    y, group_names = create_labels(fingers_angles_df, FINGER_GROUPS, JOINT_ANGLE_THRESHOLDS)
    logging.info(f"Labels created. Shape: {y.shape}")
    logging.info(f"Number of skipped samples: {skipped}")

    print_label_distribution(y, group_names, label='Processed Sessions')

    if not use_existing_scaler:
        emg_means = emg_df[emg_channels].mean()
        emg_stds = emg_df[emg_channels].std()
        return X, y, skipped, emg_means, emg_stds
    else:
        return X, y, skipped

def main():
    # Define session directories:
    # Update these paths based on your actual data directory structure
    training_session_dirs = [
        'data/EMG Hand Data 20241030_224059',
        'data/EMG Hand Data 20241030_224413',
        'data/EMG Hand Data 20241030_225347',
        'data/EMG Hand Data 20241030_231057',
        'data/EMG Hand Data 20241030_232021',
        'data/EMG Hand Data 20241030_233323',
        'data/EMG Hand Data 20241030_234704',
        'data/EMG Hand Data 20241030_235851',
        'data/EMG Hand Data 20241031_002827'
        # Add more training session directories as needed
    ]

    validation_session_dirs = [
        'data/EMG Hand Data 20241030_223524',
        # Add more validation session directories as needed
    ]

    test_session_dirs = [
        'data/EMG Hand Data 20241031_001704',
        # Add more test session directories as needed
    ]

    # Process training sessions using frame-based approach
    X_train_full, y_train_full, skipped_train, emg_means, emg_stds = process_sessions_frame_based(
        training_session_dirs,
        num_frames=NUM_FRAMES,
        use_existing_scaler=False
    )
    logging.info(f"Training data processed with {skipped_train} skipped samples.")

    # Check for invalid features in training data
    if np.isnan(X_train_full).any() or np.isinf(X_train_full).any():
        logging.error("Training features contain NaNs or infinite values. Removing invalid samples.")
        valid_indices = ~np.isnan(X_train_full).any(axis=(1,2)) & ~np.isinf(X_train_full).any(axis=(1,2))
        X_train_full = X_train_full[valid_indices]
        y_train_full = y_train_full[valid_indices]
        logging.info(f"After removal, training feature shape: {X_train_full.shape}, label shape: {y_train_full.shape}")

    # Feature scaling on training data
    # Flatten the frames to apply scaler and then reshape back
    num_samples, num_frames, num_channels = X_train_full.shape
    X_train_reshaped = X_train_full.reshape(-1, num_channels)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(num_samples, num_frames, num_channels)
    logging.info("Training feature scaling completed.")

    # Process validation sessions using training EMG normalization parameters
    X_val, y_val, skipped_val = process_sessions_frame_based(
        validation_session_dirs,
        num_frames=NUM_FRAMES,
        emg_means=emg_means,
        emg_stds=emg_stds,
        use_existing_scaler=True
    )
    logging.info(f"Validation data processed with {skipped_val} skipped samples.")

    # Check for invalid features in validation data
    if np.isnan(X_val).any() or np.isinf(X_val).any():
        logging.error("Validation features contain NaNs or infinite values. Removing invalid samples.")
        valid_indices = ~np.isnan(X_val).any(axis=(1,2)) & ~np.isinf(X_val).any(axis=(1,2))
        X_val = X_val[valid_indices]
        y_val = y_val[valid_indices]
        logging.info(f"After removal, validation feature shape: {X_val.shape}, label shape: {y_val.shape}")

    # Feature scaling on validation data
    num_val_samples = X_val.shape[0]
    X_val_reshaped = X_val.reshape(-1, num_channels)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(num_val_samples, num_frames, num_channels)
    logging.info("Validation feature scaling completed.")

    # Display label distribution
    group_names = list(FINGER_GROUPS.keys())
    print_label_distribution(y_train_full, group_names, label='Training Set')
    print_label_distribution(y_val, group_names, label='Validation Set')

    # Compute class weights for each finger group
    class_weights = {}
    for i, finger_group in enumerate(group_names):
        classes = np.unique(y_train_full[:, i])
        if len(classes) > 1:
            cw = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train_full[:, i]
            )
            class_weights[i] = dict(zip(classes, cw))
        else:
            class_weights[i] = {classes[0]: 1.0}
    logging.info(f"Computed class weights: {class_weights}")
    
    # Build the CNN model
    input_shape = (NUM_FRAMES, len(SELECTED_CHANNELS))
    output_dim = y_train_full.shape[1]
    model = build_cnn_model(input_shape, output_dim, class_weights)
    logging.info("CNN Model built.")

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    
    batch_size = 1024
    
    # Create training and validation generators
    train_generator = AugmentedDataGenerator(
        X_train_scaled, y_train_full, 
        batch_size=batch_size, 
        shuffle=True,
        add_noise=True, 
        noise_stddev=0.1,
        apply_time_warp=False, 
        max_warp=0.1,
        apply_magnitude_scaling=True, 
        scale_range=(0.25, 1.25),
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=200,  # Increased epochs for CNN
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    logging.info("Model training completed.")

    # Plot learning curves
    plot_learning_curves(history)

    # Evaluate on validation set
    evaluate_model(model, X_val_scaled, y_val, group_names, label='Validation Set')

    # Save the model and scaler
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'model_{timestamp}'
    save_model_and_preprocessing(model, scaler, emg_means, emg_stds, model_dir)

    # Process test sessions using training EMG normalization parameters
    X_test, y_test, skipped_test = process_sessions_frame_based(
        test_session_dirs,
        num_frames=NUM_FRAMES,
        emg_means=emg_means,
        emg_stds=emg_stds,
        use_existing_scaler=True
    )
    logging.info(f"Additional test data processed with {skipped_test} skipped samples.")

    # Check for invalid features in additional test set
    if np.isnan(X_test).any() or np.isinf(X_test).any():
        logging.error("Additional test features contain NaNs or infinite values. Removing invalid samples.")
        valid_indices = ~np.isnan(X_test).any(axis=(1,2)) & ~np.isinf(X_test).any(axis=(1,2))
        X_test = X_test[valid_indices]
        y_test = y_test[valid_indices]
        logging.info(f"After removal, additional test feature shape: {X_test.shape}, label shape: {y_test.shape}")

    # Feature scaling on additional test data
    num_test_samples = X_test.shape[0]
    X_test_reshaped = X_test.reshape(-1, len(SELECTED_CHANNELS))
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(num_test_samples, NUM_FRAMES, len(SELECTED_CHANNELS))
    logging.info("Additional test feature scaling completed.")

    # Evaluate on additional test set
    evaluate_model(model, X_test_scaled, y_test, group_names, label='Additional Test Set')

if __name__ == "__main__":
    main()
