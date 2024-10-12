import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Define the sampling rate of your data (in Hz)
sampling_rate = 250  # Adjust as per your data

# Load and preprocess the data
def load_and_preprocess_data(hand_file, emg_file, start_cutoff=2):
    # Load the data
    hand_df = pd.read_csv(hand_file)
    emg_df = pd.read_csv(emg_file)

    # Convert timestamps to datetime
    hand_df['timestamp_hand'] = pd.to_datetime(hand_df['timestamp_hand'], unit='s')
    emg_df['timestamp'] = pd.to_datetime(emg_df['timestamp'], unit='s')

    # Merge on nearest timestamps
    merged_df = pd.merge_asof(
        emg_df.sort_values('timestamp'),
        hand_df.sort_values('timestamp_hand'),
        left_on='timestamp',
        right_on='timestamp_hand',
        direction='nearest',
        tolerance=pd.Timedelta(milliseconds=20)  # Adjust tolerance if needed
    )

    # Drop rows where no match was found within the tolerance
    merged_df.dropna(inplace=True)

    # Filter out initial seconds
    merged_df['elapsed_seconds'] = (merged_df['timestamp'] - merged_df['timestamp'].iloc[0]).dt.total_seconds()
    merged_df = merged_df[merged_df['elapsed_seconds'] >= start_cutoff].drop(
        columns=['timestamp', 'timestamp_hand', 'elapsed_seconds']
    )

    # Define columns for normalization
    emg_columns = [col for col in emg_df.columns if 'emg_channel' in col]
    percent_columns = [col for col in hand_df.columns if '_percent' in col]

    # Normalize EMG data
    scaler_emg = MinMaxScaler()
    merged_df[emg_columns] = scaler_emg.fit_transform(merged_df[emg_columns])

    # Percent columns are already between 0 and 1 (assuming they are normalized between 0 and 1)
    # If they are between 0 and 100, normalize them to 0-1
    if merged_df[percent_columns].max().max() > 1:
        merged_df[percent_columns] = merged_df[percent_columns] / 100.0

    return merged_df, emg_columns, percent_columns

# Function to create time frames for 3-label binary classification (pinky/ring, pointer/middle, thumb)
def create_aggregated_binary_frames(
    merged_df, emg_columns, percent_columns, frame_size_ms, sampling_rate, thresholds
):
    frame_size_samples = int((frame_size_ms / 1000) * sampling_rate)

    X = merged_df[emg_columns].values

    # Apply thresholds to create binary labels
    pinky_ring = (merged_df[['pinky_percent', 'ring_percent']].mean(axis=1) < thresholds['pinky_ring']).astype(int)
    pointer_middle = (merged_df[['index_percent', 'middle_percent']].mean(axis=1) < thresholds['pointer_middle']).astype(int)
    thumb = (merged_df['thumb_percent'] > thresholds['thumb']).astype(int)

    # Stack the three groups into a single array
    y = np.vstack([pinky_ring, pointer_middle, thumb]).T

    # Create sequences
    X_frames, y_frames = [], []
    for i in range(len(X) - frame_size_samples + 1):
        X_frames.append(X[i:i + frame_size_samples])
        y_frames.append(y[i + frame_size_samples - 1])  # Use the last label in the frame

    return np.array(X_frames), np.array(y_frames)

# Define LSTM model architecture
def create_lstm_model(input_shape, output_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(output_shape, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

# Main function to execute the workflow
def main():
    hand_file = 'hand_data.csv'  # Update with your actual file name
    emg_file = 'emg_data.csv'    # Update with your actual file name
    merged_df, emg_columns, percent_columns = load_and_preprocess_data(hand_file, emg_file)

    # Define thresholds for classification
    thresholds = {
        'pinky_ring': 0.5,       # Threshold for pinky and ring fingers being down
        'pointer_middle': 0.5,   # Threshold for pointer and middle fingers being down
        'thumb': 0.5             # Threshold for thumb being open
    }

    # Create aggregated binary frames dataset for a 100ms frame size
    X_binary_frames, y_binary_frames = create_aggregated_binary_frames(
        merged_df, emg_columns, percent_columns, 100, sampling_rate, thresholds
    )

    # Reshape data to fit the model
    X_binary_frames = X_binary_frames.reshape(
        (X_binary_frames.shape[0], X_binary_frames.shape[1], len(emg_columns))
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary_frames, y_binary_frames, test_size=0.2, random_state=42
    )

    # Create the LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = y_train.shape[1]
    model = create_lstm_model(input_shape, output_shape)

    # Early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluate the model on test data
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Precision: {test_precision * 100:.2f}%")
    print(f"Test Recall: {test_recall * 100:.2f}%")

    # If needed, save the model
    model.save('emg_lstm_model.keras')

if __name__ == "__main__":
    main()
