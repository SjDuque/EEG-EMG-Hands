# realtime_prediction.py
import numpy as np
import time
from record_ema import EMARecorder, NumpyBuffer
from tensorflow.keras.models import load_model
import pylsl

class RealTimePredictor:
    def __init__(self,
                 recorder: EMARecorder,
                 model_dir: str,
                 fps: int = 30,
                 ):
        """
        Initializes the real-time predictor.

        :param recorder: Instance of EMARecorder for data collection.
        :param model_path: Path to the trained Keras model.
        """
        self.recorder = recorder
        self.model = load_model(model_dir + "/model.keras")
        self.std = np.load(model_dir + "/std.npy")
        self.ema_spans = np.load(model_dir + "/ema_spans.npy")
        self.ema_columns = np.load(model_dir + "/ema_columns.npy")
        self.finger_columns = np.load(model_dir + "/finger_columns.npy")
        self.num_fingers = 5
        self.finger_thresholds = np.load(model_dir + "/finger_thresholds.npy")
        
        # Start the lsl stream
        self.info = pylsl.StreamInfo('FingerPredictions', 'Markers', self.num_fingers, pylsl.IRREGULAR_RATE, 'float32', 'FingerPredictions')
        self.outlet = pylsl.StreamOutlet(self.info)
        
        self.fps = fps
        self.update_interval = 1 / fps
        
    def span_to_alpha(self, span):
        return 2 / (span + 1)

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
        data = self.recorder.get_data()[0]
        
        if data is None:
            return None
        
        # Get the last row of data
        ema_data = data[self.ema_columns]
        ema_data = self.preprocess_ema(ema_data).values
        
        prediction = self.model.predict(ema_data, verbose=0)
        # ema predictions
        
        # Mean of the output
        average_output = np.mean(prediction, axis=0)
        if len(average_output) == 4:
            # Add the last finger twice
            average_output = np.append(average_output, average_output[-1])
        
        binary_prediction = (average_output > 0.5).astype(int)
        
        # Stream the predictions
        self.outlet.push_sample(binary_prediction)
        # Handle the prediction (e.g., print, send over network)
        print(f'Fingers: {self.finger_columns}')
        print(f"Predictions: {binary_prediction}")

    def run_loop(self):
        """
        Starts data collection and prediction.
        """
        last_time = time.perf_counter()
        while True:
            # Start data collection
            self.recorder.update()
            # Start prediction loop
            self.predict()
            print('...')
            # Sleep to maintain the desired FPS
            current_time = time.perf_counter()
            elapsed_time = current_time - last_time
            sleep_time = self.update_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

def main():
    # Path to the trained model
    model_dir = "data/s_0/models/model_0"
    ema_spans = np.load(model_dir + "/ema_spans.npy")
    # Initialize EMARecorder
    recorder = EMARecorder(
        exg_stream_name="filtered_exg",
        angle_stream_name="FingerPercentages",
        ema_spans=list(ema_spans),
        save_angle=False,
        save_merged=False,
        save_status=False,
        save_exg=True
    )

    # Initialize RealTimePredictor
    predictor = RealTimePredictor(
        recorder=recorder,
        model_dir=model_dir,
        fps=5
    )

    # Run the real-time predictor
    predictor.run_loop()

if __name__ == "__main__":
    main()
