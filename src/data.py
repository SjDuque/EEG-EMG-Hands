import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define the sampling rate of your data (in Hz)
sampling_rate = 250  # Example: 250 Hz

# Load and preprocess the data
def load_and_preprocess_data(hand_file, emg_file, start_cutoff=2):
    # Load the data
    hand_df = pd.read_csv(hand_file)
    emg_df = pd.read_csv(emg_file)

    # Convert timestamps to datetime
    hand_df['timestamp_hand'] = pd.to_datetime(hand_df['timestamp_hand'], unit='s')
    emg_df['timestamp'] = pd.to_datetime(emg_df['timestamp'], unit='s')

    # Filter EMG channels [1, 2, 3, 4] 
    selected_channels = ['emg_channel_1', 'emg_channel_2', 'emg_channel_3', 'emg_channel_4']
    emg_df = emg_df[['timestamp'] + selected_channels]

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

    # Normalize EMG data
    scaler_emg = MinMaxScaler()
    merged_df[selected_channels] = scaler_emg.fit_transform(merged_df[selected_channels])

    # Normalize percent columns if needed
    percent_columns = [col for col in hand_df.columns if '_percent' in col]
    if merged_df[percent_columns].max().max() > 1:
        merged_df[percent_columns] = merged_df[percent_columns] / 100.0

    return merged_df, selected_channels, percent_columns

# Function to create time frames for binary classification (open/closed)
def create_binary_frames(merged_df, emg_columns, percent_columns, frame_size_ms, sampling_rate, threshold=0.5):
    frame_size_samples = int((frame_size_ms / 1000) * sampling_rate)

    X = merged_df[emg_columns].values
    y = (merged_df[percent_columns] > threshold).astype(int).values

    X_frames, y_frames = [], []
    for i in range(len(X) - frame_size_samples + 1):
        X_frames.append(X[i:i + frame_size_samples])
        y_frames.append(y[i + frame_size_samples - 1])  # Use the last frame in each segment for prediction

    return np.array(X_frames), np.array(y_frames)

# Main function to execute the workflow
def main():
    hand_file = 'hand_data.csv'
    emg_file = 'emg_data.csv'
    merged_df, emg_columns, percent_columns = load_and_preprocess_data(hand_file, emg_file)

    # Create binary frames dataset for a 100ms frame size with a 50% threshold
    X_binary_frames, y_binary_frames = create_binary_frames(merged_df, emg_columns, percent_columns, 100, sampling_rate, threshold=0.5)
    print("Binary data frames shape:", X_binary_frames.shape, y_binary_frames.shape)
    
    # Verify binary label distribution for each finger
    for i in range(y_binary_frames.shape[1]):
        unique, counts = np.unique(y_binary_frames[:, i], return_counts=True)
        print(f"Finger {i+1} class distribution:", dict(zip(unique, counts)))

if __name__ == "__main__":
    main()
