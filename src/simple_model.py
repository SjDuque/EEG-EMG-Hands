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
    """
    Save the trained model, preprocessing scaler, and EMG normalization parameters to disk.
    """
    # Create the directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    # Save the model
    model.save(os.path.join(model_dir, 'model.keras'))
    # Save the feature scaler
    joblib.dump(scaler, os.path.join(model_dir, 'feature_scaler.joblib'))
    # Save the EMG normalization parameters
    joblib.dump({'means': emg_means, 'stds': emg_stds}, os.path.join(model_dir, 'emg_normalization_params.joblib'))
    logging.info(f"Model and preprocessing saved to directory {model_dir}.")

# Configuration Parameters
SAMPLING_RATE = 250  # in Hz
SELECTED_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]  # EMG channels to use
FRAME_SIZES = [1, 2, 4, 8, 16, 32, 48, 64]  # Frame sizes in samples

# Thresholds for determining if a finger is closed (midpoints)
JOINT_ANGLE_THRESHOLDS = {
    'thumb':    (132.5  + 157.5)    / 2,    # (132.5 + 157.5) / 2
    'index':    (60     + 160)      / 2,     # (25 + 165) / 2
    'middle':   (40     + 167.5)    / 2,   # (15 + 167.5) / 2
    'ring':     (30     + 167.5)    / 2,     # (15 + 167.5) / 2
    'pinky':    (30     + 167.5)    / 2   # (15 + 167.5) / 2
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def find_nearest_previous_timestamp(emg_timestamps, target_timestamp):
    """
    Find the index of the nearest previous EMG timestamp for a given finger timestamp.
    """
    indices = np.where(emg_timestamps <= target_timestamp)[0]
    if len(indices) == 0:
        return None
    return indices[-1]

def extract_emg_features(emg_data, frame_size_samples):
    """
    Extract statistical features from EMG data for a given frame size.
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

def create_labels(fingers_angles_df):
    """
    Create binary labels indicating whether each finger group is closed.
    """
    labels = []
    for _, row in fingers_angles_df.iterrows():
        thumb_label = int(row['THUMB'] < JOINT_ANGLE_THRESHOLDS['thumb'])
        # Group Index and Middle fingers
        index_middle_label = int(
            (row['INDEX'] < JOINT_ANGLE_THRESHOLDS['index']) or
            (row['MIDDLE'] < JOINT_ANGLE_THRESHOLDS['middle'])
        )
        # Group Ring and Pinky fingers
        ring_pinky_label = int(
            (row['RING'] < JOINT_ANGLE_THRESHOLDS['ring']) or
            (row['PINKY'] < JOINT_ANGLE_THRESHOLDS['pinky'])
        )
        label = [thumb_label, index_middle_label, ring_pinky_label]
        labels.append(label)
    return np.array(labels)

def print_label_distribution(y, label='Overall'):
    """
    Print the distribution of labels for each finger group.
    """
    total_samples = y.shape[0]
    label_sums = np.sum(y, axis=0)
    label_counts = {
        'Thumb': label_sums[0],
        'Index_Middle': label_sums[1],
        'Ring_Pinky': label_sums[2]
    }
    print(f"\nLabel distribution ({label}):")
    for finger_group, count in label_counts.items():
        print(f"{finger_group}: {count}/{total_samples} ({(count / total_samples) * 100:.2f}%)")

def build_model(input_dim, output_dim):
    """
    Build and compile a neural network model.
    """
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
        Dense(output_dim, activation='sigmoid')  # Sigmoid for multi-label classification
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

def plot_learning_curves(history):
    """
    Plot training and validation binary accuracy and loss.
    """
    plt.figure(figsize=(12, 5))

    # Binary Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['binary_accuracy'], label='Train Binary Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Binary Accuracy')
    plt.title('Model Binary Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Accuracy')
    plt.legend()

    # Loss Plot
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
    # File paths
    emg_file = 'EMG Hand Data 20241031_002827/emg.csv'
    fingers_file = 'EMG Hand Data 20241031_002827/finger_angles.csv'

    # Load EMG data
    try:
        emg_df = pd.read_csv(emg_file)
        fingers_df = pd.read_csv(fingers_file)
        logging.info("Data files loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return

    # Check for required columns in EMG data
    expected_emg_columns = ['timestamp'] + [f'emg_channel_{ch}' for ch in SELECTED_CHANNELS]
    if not all(col in emg_df.columns for col in expected_emg_columns):
        logging.error("EMG data file is missing required columns.")
        return

    # Check for required columns in finger data
    expected_finger_columns = ['timestamp', 'THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']  # Corrected 'PINKY' instead of duplicate 'THUMB'
    if not all(col in fingers_df.columns for col in expected_finger_columns):
        logging.error("Finger data file is missing required columns.")
        return

    # Drop any NaN values in finger data for angle columns
    fingers_df.dropna(subset=expected_finger_columns, inplace=True)
    emg_df.dropna(subset=expected_emg_columns, inplace=True)

    # Compute means and stds per channel for normalization
    emg_channels = [f'emg_channel_{ch}' for ch in SELECTED_CHANNELS]
    emg_means = emg_df[emg_channels].mean()
    emg_stds = emg_df[emg_channels].std()
    
    # Normalize EMG data per channel using training data stats
    emg_df[emg_channels] = (emg_df[emg_channels] - emg_means) / emg_stds
    logging.info("EMG data normalized.")

    # Extract EMG channels and timestamps
    emg_data = emg_df[emg_channels].values
    emg_timestamps = emg_df['timestamp'].values
    logging.info("EMG data extracted.")

    # Extract finger timestamps and angles
    fingers_timestamps = fingers_df['timestamp'].values
    fingers_angles = fingers_df[expected_finger_columns[1:]]  # ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']
    logging.info("Finger data extracted.")

    # Initialize feature list and collect corresponding finger angles
    feature_list = []
    corresponding_finger_angles = []
    skipped = 0

    for idx, finger_timestamp in enumerate(fingers_timestamps):
        nearest_emg_idx = find_nearest_previous_timestamp(emg_timestamps, finger_timestamp)
        if nearest_emg_idx is None:
            # If no previous EMG data, skip this finger timestamp
            logging.warning(f"No EMG data before finger timestamp {finger_timestamp}. Skipping.")
            skipped += 1
            continue

        features = []
        for frame_size in FRAME_SIZES:
            start_idx = max(0, nearest_emg_idx - frame_size + 1)
            end_idx = nearest_emg_idx + 1  # Include the nearest sample

            frame_data = emg_data[start_idx:end_idx, :]
            frame_features = extract_emg_features(frame_data, frame_size)
            features.extend(frame_features)

        feature_list.append(features)
        # Collect the corresponding finger angles for label creation
        corresponding_finger_angles.append(fingers_angles.iloc[idx])

    # Convert feature list to numpy array
    X = np.array(feature_list)
    logging.info(f"Feature extraction completed. Shape: {X.shape}")

    # Convert collected finger angles to DataFrame
    fingers_angles_df = pd.DataFrame(corresponding_finger_angles)

    # Create labels using the create_labels function
    y = create_labels(fingers_angles_df)
    logging.info(f"Labels created. Shape: {y.shape}")
    logging.info(f"Number of skipped samples: {skipped}")

    # Print label distribution for the entire dataset
    print_label_distribution(y, label='Overall')

    # Check for any NaNs or infinite values in features
    if np.isnan(X).any() or np.isinf(X).any():
        logging.error("Features contain NaNs or infinite values. Please check data preprocessing.")
        # Remove such samples
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        logging.info(f"After removing invalid samples, feature shape: {X.shape}, label shape: {y.shape}")

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Feature scaling completed.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets.")

    # Print label distributions for training and testing sets
    print_label_distribution(y_train, label='Training Set')
    print_label_distribution(y_test, label='Test Set')

    # Compute class weights for each finger group to handle class imbalance
    class_weights = {}
    for i, finger_group in enumerate(['Thumb', 'Index_Middle', 'Ring_Pinky']):
        classes = np.unique(y_train[:, i])
        if len(classes) > 1:
            cw = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train[:, i]
            )
            class_weights[i] = dict(zip(classes, cw))
        else:
            # If only one class present, assign weight 1
            class_weights[i] = {classes[0]: 1.0}
    logging.info(f"Computed class weights: {class_weights}")
    # Note: Keras does not directly support class weights for multi-output. You might need to handle this manually or adjust the loss function accordingly.

    # Build the model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = build_model(input_dim, output_dim)
    logging.info("Model built.")

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    logging.info("Model training completed.")

    # Plot learning curves
    plot_learning_curves(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    logging.info(f"Test Loss: {loss:.4f}")

    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Thumb', 'Index_Middle', 'Ring_Pinky'],
        zero_division=0
    ))

    # Print confusion matrix for each class
    for i, finger_group in enumerate(['Thumb', 'Index_Middle', 'Ring_Pinky']):
        print(f"\nConfusion Matrix for {finger_group}:")
        cm = confusion_matrix(y_test[:, i], y_pred[:, i])
        print(cm)
        
    # Save the model and preprocessing 
    # Use timestamp to differentiate between models
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'model_{timestamp}'
    save_model_and_preprocessing(model, scaler, emg_means, emg_stds, model_dir)

if __name__ == "__main__":
    main()
