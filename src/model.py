# model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Import the required functions from data.py
from data import (
    load_and_preprocess_data, 
    create_aggregated_binary_frames, 
    augment_data, 
    save_model_and_preprocessing
)

# Set random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def create_lstm_model(input_shape, output_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_shape, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )
    return model

# Learning rate scheduler
def scheduler(epoch, lr):
    return lr if epoch < 10 else lr * np.exp(-0.1)

# Main function to execute the workflow
def main():
    set_seeds()

    # Define file paths
    hand_file = 'fingers_1.csv'
    emg_file = 'emg_1.csv'

    # Load and preprocess the data
    hand_df, emg_df, scaler_emg = load_and_preprocess_data(hand_file, emg_file)

    # Create aggregated binary frames dataset
    X_binary_frames, y_binary_frames = create_aggregated_binary_frames(
        hand_df, emg_df
    )

    # Analyze label distribution
    label_sums = y_binary_frames.sum(axis=0)
    print(f"Label distribution: {label_sums}")

    # Split data into training and testing sets
    y_combined = np.array([''.join(map(str, row.astype(int))) for row in y_binary_frames])
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary_frames, y_binary_frames, test_size=0.2, random_state=42, stratify=y_combined
    )

    # Data augmentation
    X_train_aug, y_train_aug = augment_data(X_train, y_train)

    # Shuffle augmented data
    idx = np.arange(len(X_train_aug))
    np.random.shuffle(idx)
    X_train_aug = X_train_aug[idx]
    y_train_aug = y_train_aug[idx]

    # Create the LSTM model
    input_shape = (X_train_aug.shape[1], X_train_aug.shape[2])
    output_shape = y_train_aug.shape[1]
    model = create_lstm_model(input_shape, output_shape)

    # Early stopping and learning rate scheduler
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # Train the model
    history = model.fit(
        X_train_aug, y_train_aug,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler]
    )

    # Evaluate the model on test data
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Precision: {test_precision * 100:.2f}%")
    print(f"Test Recall: {test_recall * 100:.2f}%")

    # Save the model and preprocessing information
    save_model_and_preprocessing(model, scaler_emg, thresholds)

if __name__ == "__main__":
    main()
