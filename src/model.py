import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os

def save_model_and_preprocessing(model, scaler, model_dir):
    """
    Saves the trained model and the scaler to the specified directory.

    Parameters:
    - model: Trained Keras model.
    - scaler: Fitted StandardScaler object.
    - model_dir: Directory path to save the model and scaler.
    """
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'model.keras'))
    joblib.dump(scaler, os.path.join(model_dir, 'feature_scaler.joblib'))
    logging.info(f"Model and scaler saved to directory {model_dir}.")

# Configuration Parameters
SAMPLING_RATE = 250
SELECTED_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]
FINGER_GROUPS = {
    'Thumb': ['THUMB'],
    'Index_Middle': ['INDEX', 'MIDDLE'],
    'Ring_Pinky': ['RING', 'PINKY'],
}

JOINT_ANGLE_THRESHOLDS = {
    'THUMB':  (132.5 + 157.5) / 2,
    'INDEX':  (60 + 160) / 2,
    'MIDDLE': (40 + 167.5) / 2,
    'RING':   (30 + 167.5) / 2,
    'PINKY':  (30 + 167.5) / 2
}

# New Configuration Parameters for Frame-Based Processing
NUM_FRAMES = 50  # Number of frames to include before the timestamp
FRAME_STEP = 1   # Step size between frames

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def find_nearest_previous_timestamp(emg_timestamps, target_timestamp):
    """
    Finds the index of the nearest EMG timestamp that is less than or equal to the target timestamp.

    Parameters:
    - emg_timestamps: Numpy array of EMG timestamps.
    - target_timestamp: The target timestamp to find the nearest previous EMG timestamp for.

    Returns:
    - Index of the nearest previous timestamp or None if not found.
    """
    idx = np.searchsorted(emg_timestamps, target_timestamp, side='right') - 1
    return idx if idx >= 0 else None

def create_labels(fingers_angles_df, finger_groups, thresholds):
    """
    Vectorized creation of binary labels indicating whether each finger group is closed for each sample.

    Parameters:
    - fingers_angles_df: DataFrame with finger angles for each sample.
    - finger_groups: Dict mapping group names to list of fingers.
    - thresholds: Dict mapping individual fingers to their threshold.

    Returns:
    - labels: numpy array with binary labels per group for each sample.
    - group_names: List of group names in the order of labels.
    """
    labels = np.zeros((len(fingers_angles_df), len(finger_groups)), dtype=int)
    group_names = list(finger_groups.keys())

    # Iterate over each finger group
    for i, (group_name, fingers) in enumerate(finger_groups.items()):
        # Determine if any finger in the group is below its threshold
        group_condition = np.any(
            fingers_angles_df[fingers].lt([thresholds[f] for f in fingers], axis=1),
            axis=1
        )
        labels[:, i] = group_condition.astype(int)

    return labels, group_names

def print_label_distribution(y, group_names, label='Overall'):
    """
    Print the distribution of labels for each finger group.

    Parameters:
    - y: numpy array with binary labels per group for each sample.
    - group_names: List of group names corresponding to the labels.
    - label: Descriptor for the label distribution being printed (e.g., 'Overall', 'Training Set').
    """
    total_samples = y.shape[0]
    label_sums = np.sum(y, axis=0)
    print(f"\nLabel distribution ({label}):")
    for i, finger_group in enumerate(group_names):
        count = label_sums[i]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"{finger_group}: {count}/{total_samples} ({percentage:.2f}%)")

def build_cnn_model(input_shape, output_dim):
    """
    Builds and compiles the CNN neural network model.

    Parameters:
    - input_shape: Tuple representing the shape of the input data (NUM_FRAMES, NUM_CHANNELS).
    - output_dim: Number of output classes.

    Returns:
    - Compiled Keras model.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.3),

        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(output_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

def plot_learning_curves(history):
    """
    Plots the learning curves for accuracy and loss.

    Parameters:
    - history: Keras History object from model training.
    """
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

def evaluate_model(model, X, y, group_names, label='Test Set'):
    """
    Evaluate the model and print classification metrics.

    Parameters:
    - model: Trained Keras model.
    - X: Features for evaluation.
    - y: True labels for evaluation.
    - group_names: Names of the finger groups.
    - label: Descriptor for the evaluation set.
    """
    loss, accuracy = model.evaluate(X, y, verbose=0)
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

def main():
    # Define session directories
    training_session_dirs = [
        'EMG Hand Data 20241030_223524',
        'EMG Hand Data 20241030_224059',
        'EMG Hand Data 20241030_224413',
        'EMG Hand Data 20241030_225347',
        'EMG Hand Data 20241030_231057',
        'EMG Hand Data 20241030_232021',
        'EMG Hand Data 20241030_233323',
        'EMG Hand Data 20241030_234704',
        'EMG Hand Data 20241030_235851',
        # Add more training session directories as needed
    ]

    validation_session_dirs = [
        'EMG Hand Data 20241031_001704',
        # Add more validation session directories as needed
    ]

    test_session_dirs = [
        'EMG Hand Data 20241031_002827',
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
    model = build_cnn_model(input_shape, output_dim)
    logging.info("CNN Model built.")

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        X_train_scaled, y_train_full,
        epochs=5,  # Increased epochs for CNN
        batch_size=256,  # Adjusted batch size
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
    save_model_and_preprocessing(model, scaler, model_dir)

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
