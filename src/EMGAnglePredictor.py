import os
import time
import numpy as np
import pandas as pd
import cv2
from HandTracking import HandTracking
from pylsl import StreamInlet, resolve_stream
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class EMGAnglePredictor:
    def __init__(self, model_path='emg_angle_predictor_model.h5'):
        self.model_path = model_path
        
        # Load or create the model
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print("Pre-trained model loaded.")
        else:
            print("No pre-trained model found. Collecting training data...")
            self.model = None

        # Resolve the EMG stream
        print("Looking for an EMG stream...")
        self.streams = resolve_stream('type', 'EMG')
        self.inlet = StreamInlet(self.streams[0])
        print("EMG stream found!")

        # Initialize HandTracking
        self.hand_tracker = HandTracking()

    def collect_training_data(self, duration=60):
        """Collect training data for the specified duration."""
        start_time = time.time()
        data = []

        self.hand_tracker.start()

        try:
            while time.time() - start_time < duration:
                success, image = self.hand_tracker.cap.read()
                if not success:
                    print("Failed to capture image.")
                    continue

                # Flip the image horizontally for a later selfie-view display
                image = cv2.flip(image, 1)
                # Convert the BGR image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Process the image and detect hands
                results = self.hand_tracker.hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        angles, percentages = self.hand_tracker.get_calculated_angles(hand_landmarks)

                        # Collect EMG data sample
                        sample, timestamp = self.inlet.pull_sample()
                        emg_sample = sample  # Replace with actual EMG data

                        # Append the data sample with angles
                        sample = emg_sample + list(angles.values())
                        data.append(sample)

                cv2.waitKey(1)
        finally:
            self.hand_tracker.close()

        # Save data to a DataFrame
        columns = ['emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8', 'thumb', 'index', 'middle', 'ring', 'pinky']
        df = pd.DataFrame(data, columns=columns)

        return df

    def create_and_train_model(self, data):
        """Create and train a new model using the provided training data."""
        # Separate features and targets
        X = data[['emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8']].values
        y = data[['thumb', 'index', 'middle', 'ring', 'pinky']].values

        # Split the data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the model
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(5, activation='linear'))  # Output layer for 5 angles

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss}')

        # Save the model
        model.save(self.model_path)
        print("Model trained and saved.")

        return model

    def start(self):
        """Start the video capture and hand tracking."""
        self.hand_tracker.start()

    def close(self):
        """Close the hand tracking and release resources."""
        self.hand_tracker.close()

    def is_opened(self):
        """Check if the hand tracking video capture is opened."""
        return self.hand_tracker.isOpened()

    def predict_angles_from_emg(self):
        """Predict angles using real-time EMG data."""
        sample, timestamp = self.inlet.pull_sample()
        emg_sample = np.array(sample).reshape(1, -1)
        predicted_angles = self.model.predict(emg_sample)
        return {finger: angle for finger, angle in zip(self.hand_tracker.joint_sets.keys(), predicted_angles[0])}

    def update(self, draw_camera=True, draw_angles=True, terminal_out=False):
        """Update the hand tracking and display results with predicted angles."""
        # Get predicted angles from EMG data
        predicted_angles = self.predict_angles_from_emg()

        # Update the hand tracker with the predicted angles
        self.hand_tracker.update(draw_camera=draw_camera, draw_angles=draw_angles, terminal_out=terminal_out, predicted_angles=predicted_angles)

def main():
    emg_predictor = EMGAnglePredictor(model_path='emg_angle_predictor_model.h5')

    if emg_predictor.model is None:
        # Collect training data and train the model if no pre-trained model is available
        training_data = emg_predictor.collect_training_data(duration=60)
        emg_predictor.model = emg_predictor.create_and_train_model(training_data)

    emg_predictor.start()

    try:
        while emg_predictor.is_opened():
            emg_predictor.update(draw_camera=True, draw_angles=True, terminal_out=True)
    finally:
        emg_predictor.close()

if __name__ == "__main__":
    main()
