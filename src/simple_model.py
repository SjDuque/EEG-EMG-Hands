import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os

def save_model_and_preprocessing(model, scaler, emg_means, emg_stds, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'model.keras'))
    joblib.dump(scaler, os.path.join(model_dir, 'feature_scaler.joblib'))
    joblib.dump({'means': emg_means, 'stds': emg_stds}, os.path.join(model_dir, 'emg_normalization_params.joblib'))
    logging.info(f"Model and preprocessing saved to directory {model_dir}.")

# Configuration Parameters
SAMPLING_RATE = 250
SELECTED_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]
FRAME_SIZES = [1, 2, 4, 8, 16, 32, 48, 64]

FINGER_GROUPS = {
    'Thumb': ['THUMB'],
    'Index': ['INDEX'],
    'Middle': ['MIDDLE'],
    'Ring': ['RING'],
    'Pinky': ['PINKY'],
}

JOINT_ANGLE_THRESHOLDS = {
    'THUMB':  (132.5 + 157.5) / 2,
    'INDEX':  (60 + 160) / 2,
    'MIDDLE': (40 + 167.5) / 2,
    'RING':   (30 + 167.5) / 2,
    'PINKY':  (30 + 167.5) / 2
}

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def find_nearest_previous_timestamp(emg_timestamps, target_timestamp):
    idx = np.searchsorted(emg_timestamps, target_timestamp, side='right') - 1
    return idx if idx >= 0 else None

def extract_emg_features(emg_data, frame_size_samples):
    """
    Vectorized feature extraction.
    
    Parameters:
    - emg_data: numpy array of shape (num_samples, num_channels)
    - frame_size_samples: Number of samples in each frame
    
    Returns:
    - features: A 1D numpy array of features per frame.
    """
    # Mean Absolute Value (MAV)
    mav = np.mean(np.abs(emg_data), axis=0)

    # Root Mean Square (RMS)
    rms = np.sqrt(np.mean(emg_data ** 2, axis=0))

    # Variance (VAR)
    var = np.var(emg_data, axis=0)

    # Standard Deviation (STD)
    std = np.std(emg_data, axis=0)

    # Zero Crossing Rate (ZC)
    zc = np.sum(np.diff(np.sign(emg_data), axis=0) != 0, axis=0)

    # Slope Sign Change (SSC)
    ssc = np.sum(np.diff(np.sign(np.diff(emg_data, axis=0)), axis=0) != 0, axis=0)

    # Waveform Length (WL)
    wl = np.sum(np.abs(np.diff(emg_data, axis=0)), axis=0)

    # Concatenate all features into a single vector
    features = np.concatenate([mav, rms, var, std, zc, ssc, wl])

    return features

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

def build_model(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(output_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
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

def main():
    session_dirs = [
        'EMG Hand Data 20241030_223524',
        'EMG Hand Data 20241030_224059',
        'EMG Hand Data 20241030_224413',
        'EMG Hand Data 20241030_232021',
        'EMG Hand Data 20241030_233323',
        'EMG Hand Data 20241030_234704',
        'EMG Hand Data 20241030_235851',
        'EMG Hand Data 20241031_001704',
        'EMG Hand Data 20241031_002827',
        # Add more session directories as needed
    ]

    test_session_dirs = [
        'EMG Hand Data 20241030_225347',
        'EMG Hand Data 20241030_231057',
        # Add more test session directories as needed
    ]

    all_features = []
    all_labels = []
    skipped = 0

    all_emg_data = []

    for session_dir in session_dirs:
        emg_file = os.path.join(session_dir, 'emg.csv')
        try:
            emg_df = pd.read_csv(emg_file)
            emg_channels = [f'emg_channel_{ch}' for ch in SELECTED_CHANNELS]
            all_emg_data.append(emg_df[emg_channels])
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            continue

    all_emg_data = pd.concat(all_emg_data)
    emg_means = all_emg_data.mean()
    emg_stds = all_emg_data.std()

    for session_dir in session_dirs:
        emg_file = os.path.join(session_dir, 'emg.csv')
        fingers_file = os.path.join(session_dir, 'finger_angles.csv')

        try:
            emg_df = pd.read_csv(emg_file)
            fingers_df = pd.read_csv(fingers_file)
            logging.info(f"Data files loaded successfully for session {session_dir}.")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
        expected_emg_columns = ['timestamp'] + [f'emg_channel_{ch}' for ch in SELECTED_CHANNELS]
        if not all(col in emg_df.columns for col in expected_emg_columns):
            logging.error("EMG data file is missing required columns.")
            continue

        expected_finger_columns = ['timestamp'] + list(JOINT_ANGLE_THRESHOLDS.keys())
        if not all(col in fingers_df.columns for col in expected_finger_columns):
            logging.error("Finger data file is missing required columns.")
            continue

        fingers_df.dropna(subset=expected_finger_columns, inplace=True)
        emg_df.dropna(subset=expected_emg_columns, inplace=True)

        emg_channels = [f'emg_channel_{ch}' for ch in SELECTED_CHANNELS]
        emg_means = emg_df[emg_channels].mean()
        emg_stds = emg_df[emg_channels].std()
        
        emg_df[emg_channels] = (emg_df[emg_channels] - emg_means) / emg_stds
        logging.info("EMG data normalized.")

        emg_data = emg_df[emg_channels].values
        emg_timestamps = emg_df['timestamp'].values
        logging.info("EMG data extracted.")

        fingers_timestamps = fingers_df['timestamp'].values
        fingers_angles = fingers_df[expected_finger_columns[1:]]
        logging.info("Finger data extracted.")

        for idx, finger_timestamp in enumerate(fingers_timestamps):
            nearest_emg_idx = find_nearest_previous_timestamp(emg_timestamps, finger_timestamp)
            if nearest_emg_idx is None:
                logging.warning(f"No EMG data before finger timestamp {finger_timestamp}. Skipping.")
                skipped += 1
                continue

            features = []
            for frame_size in FRAME_SIZES:
                start_idx = max(0, nearest_emg_idx - frame_size + 1)
                end_idx = nearest_emg_idx + 1

                frame_data = emg_data[start_idx:end_idx, :]
                frame_features = extract_emg_features(frame_data, frame_size)
                features.extend(frame_features)

            all_features.append(features)
            all_labels.append(fingers_angles.iloc[idx])

    X = np.array(all_features)
    logging.info(f"Feature extraction completed. Shape: {X.shape}")

    fingers_angles_df = pd.DataFrame(all_labels)

    y, group_names = create_labels(fingers_angles_df, FINGER_GROUPS, JOINT_ANGLE_THRESHOLDS)
    logging.info(f"Labels created. Shape: {y.shape}")
    logging.info(f"Number of skipped samples: {skipped}")

    print_label_distribution(y, group_names, label='Overall')

    if np.isnan(X).any() or np.isinf(X).any():
        logging.error("Features contain NaNs or infinite values. Please check data preprocessing.")
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        logging.info(f"After removing invalid samples, feature shape: {X.shape}, label shape: {y.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Feature scaling completed.")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets.")

    print_label_distribution(y_train, group_names, label='Training Set')
    print_label_distribution(y_test, group_names, label='Test Set')

    class_weights = {}
    for i, finger_group in enumerate(group_names):
        classes = np.unique(y_train[:, i])
        if len(classes) > 1:
            cw = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train[:, i]
            )
            class_weights[i] = dict(zip(classes, cw))
        else:
            class_weights[i] = {classes[0]: 1.0}
    logging.info(f"Computed class weights: {class_weights}")

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = build_model(input_dim, output_dim)
    logging.info("Model built.")

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    logging.info("Model training completed.")

    plot_learning_curves(history)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    logging.info(f"Test Loss: {loss:.4f}")

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=group_names,
        zero_division=0
    ))

    for i, finger_group in enumerate(group_names):
        print(f"\nConfusion Matrix for {finger_group}:")
        cm = confusion_matrix(y_test[:, i], y_pred[:, i])
        print(cm)
        
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'model_{timestamp}'
    save_model_and_preprocessing(model, scaler, emg_means, emg_stds, model_dir)

    # Load and test on additional test sessions
    test_features = []
    test_labels = []
    test_skipped = 0

    for session_dir in test_session_dirs:
        emg_file = os.path.join(session_dir, 'emg.csv')
        fingers_file = os.path.join(session_dir, 'finger_angles.csv')

        try:
            emg_df = pd.read_csv(emg_file)
            fingers_df = pd.read_csv(fingers_file)
            logging.info(f"Data files loaded successfully for test session {session_dir}.")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            continue

        expected_emg_columns = ['timestamp'] + [f'emg_channel_{ch}' for ch in SELECTED_CHANNELS]
        if not all(col in emg_df.columns for col in expected_emg_columns):
            logging.error("EMG data file is missing required columns.")
            continue

        expected_finger_columns = ['timestamp'] + list(JOINT_ANGLE_THRESHOLDS.keys())
        if not all(col in fingers_df.columns for col in expected_finger_columns):
            logging.error("Finger data file is missing required columns.")
            continue

        fingers_df.dropna(subset=expected_finger_columns, inplace=True)
        emg_df.dropna(subset=expected_emg_columns, inplace=True)

        emg_df[emg_channels] = (emg_df[emg_channels] - emg_means) / emg_stds
        logging.info("EMG data normalized.")

        emg_data = emg_df[emg_channels].values
        emg_timestamps = emg_df['timestamp'].values
        logging.info("EMG data extracted.")

        fingers_timestamps = fingers_df['timestamp'].values
        fingers_angles = fingers_df[expected_finger_columns[1:]]
        logging.info("Finger data extracted.")

        for idx, finger_timestamp in enumerate(fingers_timestamps):
            nearest_emg_idx = find_nearest_previous_timestamp(emg_timestamps, finger_timestamp)
            if nearest_emg_idx is None:
                logging.warning(f"No EMG data before finger timestamp {finger_timestamp}. Skipping.")
                test_skipped += 1
                continue

            features = []
            for frame_size in FRAME_SIZES:
                start_idx = max(0, nearest_emg_idx - frame_size + 1)
                end_idx = nearest_emg_idx + 1

                frame_data = emg_data[start_idx:end_idx, :]
                frame_features = extract_emg_features(frame_data, frame_size)
                features.extend(frame_features)

            test_features.append(features)
            test_labels.append(fingers_angles.iloc[idx])

    X_test_additional = np.array(test_features)
    logging.info(f"Additional test feature extraction completed. Shape: {X_test_additional.shape}")

    fingers_angles_df_test = pd.DataFrame(test_labels)

    y_test_additional, _ = create_labels(fingers_angles_df_test, FINGER_GROUPS, JOINT_ANGLE_THRESHOLDS)
    logging.info(f"Additional test labels created. Shape: {y_test_additional.shape}")
    logging.info(f"Number of skipped samples in additional test set: {test_skipped}")

    if np.isnan(X_test_additional).any() or np.isinf(X_test_additional).any():
        logging.error("Additional test features contain NaNs or infinite values. Please check data preprocessing.")
        valid_indices = ~np.isnan(X_test_additional).any(axis=1) & ~np.isinf(X_test_additional).any(axis=1)
        X_test_additional = X_test_additional[valid_indices]
        y_test_additional = y_test_additional[valid_indices]
        logging.info(f"After removing invalid samples, additional test feature shape: {X_test_additional.shape}, label shape: {y_test_additional.shape}")

    X_test_additional_scaled = scaler.transform(X_test_additional)
    logging.info("Additional test feature scaling completed.")

    loss_additional, accuracy_additional = model.evaluate(X_test_additional_scaled, y_test_additional, verbose=0)
    logging.info(f"Additional Test Accuracy: {accuracy_additional * 100:.2f}%")
    logging.info(f"Additional Test Loss: {loss_additional:.4f}")

    y_pred_prob_additional = model.predict(X_test_additional_scaled)
    y_pred_additional = (y_pred_prob_additional > 0.5).astype(int)

    print("\nAdditional Test Classification Report:")
    print(classification_report(
        y_test_additional, y_pred_additional,
        target_names=group_names,
        zero_division=0
    ))

    for i, finger_group in enumerate(group_names):
        print(f"\nConfusion Matrix for {finger_group} (Additional Test):")
        cm = confusion_matrix(y_test_additional[:, i], y_pred_additional[:, i])
        print(cm)

if __name__ == "__main__":
    main()
