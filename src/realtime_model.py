# realtime_prediction.py
import numpy as np
import time
import threading
from record_ema import EMARecorder, NumpyBuffer
from tensorflow.keras.models import load_model

class RealTimePredictor:
    def __init__(self,
                 recorder: EMARecorder,
                 model_dir: str,
                 ):
        """
        Initializes the real-time predictor.

        :param recorder: Instance of EMARecorder for data collection.
        :param model_path: Path to the trained Keras model.
        """
        self.recorder = recorder
        self.model = load_model(model_dir + "/model.keras")
        self.std = np.load(model_dir + "/std.npy")
        self.ema_columns = np.load(model_dir + "/ema_columns.npy")
        self.finger_columns = np.load(model_dir + "/finger_columns.npy")
        self.finger_thresholds = np.load(model_dir + "/finger_thresholds.npy")

    def preprocess_ema(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocesses the data by normalizing.

        :param data: Raw EMA data.
        :return: Normalized data.
        """
        # Normalize using pre-loaded mean and std
        normalized = data / self.std
        return normalized
    
    def preprocess_fingers(self, data, finger_columns, finger_thresholds):
        """Preprocesses the finger angle data by normalizing it."""
        data = data.copy()
        for i, finger in enumerate(finger_columns):
            finger_threshold = finger_thresholds[i]
            data[finger] = (data[finger] - finger_threshold[0]) / (finger_threshold[1] - finger_threshold[0])
            data[finger] = data[finger].clip(0, 1)
            
        return data

    def predict(self):
        """
        Continuously checks the buffer for new data and makes predictions.
        """
        data = self.recorder.get_data()
        if data is None:
            return None
        
        # Get the last row of data
        ema_data = data[self.ema_columns]
        ema_data = self.preprocess_ema(ema_data).values
        finger_data = data[self.finger_columns]
        finger_data = self.preprocess_fingers(finger_data, self.finger_columns, self.finger_thresholds).values
        prediction = self.model.predict(ema_data[-1].reshape(1, -1), verbose=0)
        binary_prediction = (prediction > 0.5).astype(int)
        # Handle the prediction (e.g., print, send over network)
        print(f'Fingers: {self.finger_columns}')
        print(f"Predictions: {binary_prediction[-1]}")
        
        # # Train the model
        # if np.any(binary_prediction):
        #     self.model.fit(ema_data, finger_data, verbose=0)

    def run_loop(self):
        """
        Starts data collection and prediction.
        """
        while True:
            # Start data collection
            self.recorder.update()
            # Start prediction loop
            self.predict()
            
            print('...')
            time.sleep(0.1)

def main():
    # Initialize EMARecorder
    recorder = EMARecorder(
        exg_stream_name="filtered_exg",
        angle_stream_name="FingerPercentages",
        csv_filename="data/realtime_ema_data",
        ema_spans=[1, 2, 4, 8, 16, 32, 64]
    )

    # Path to the trained model
    model_dir = "models/s_4/model_0"

    # Initialize RealTimePredictor
    predictor = RealTimePredictor(
        recorder=recorder,
        model_dir=model_dir
    )

    # Run the real-time predictor
    predictor.run_loop()

if __name__ == "__main__":
    main()
