# realtime_prediction.py
import numpy as np
import time
from exg.ema import EMA
from tensorflow.keras.models import load_model
import pylsl

class RealTimePredictor:
    def __init__(self,
                 filtered_exg_stream: str,
                 model_dir: str,
                 fps: int = 30,
                 ):
        """
        Initializes the real-time predictor.

        :param recorder: Instance of EMARecorder for data collection.
        :param model_path: Path to the trained Keras model.
        """
        self.model = load_model(model_dir + "/model.keras")
        self.std = np.load(model_dir + "/std.npy")
        self.window_sizes = np.load(model_dir + "/window_sizes.npy")
        self.ema_columns = np.load(model_dir + "/ema_columns.npy", allow_pickle=True)
        self.finger_columns = np.load(model_dir + "/finger_columns.npy", allow_pickle=True)
        self.num_fingers = 5
        self.prev_average_output = np.zeros(self.num_fingers)
        
        self.ema_processor = EMA(
            window_sizes=self.window_sizes,
            num_channels=16,
            fs=125
        )
        self._proc_flags = (
            pylsl.proc_clocksync |
            pylsl.proc_dejitter |
            pylsl.proc_monotonize
        )
        
        # Initialize the EXG stream
        # Resolve and open LSL inlet
        found = pylsl.resolve_byprop('name', filtered_exg_stream, timeout=5)
        if not found:
            raise RuntimeError(f"No LSL stream named '{filtered_exg_stream}' found.")
        self.exg_inlet = pylsl.StreamInlet(
            found[0], max_buflen=1024, processing_flags=self._proc_flags
        )
        
        # Start the lsl stream
        self.info = pylsl.StreamInfo('finger_predictions', 'Markers', self.num_fingers, pylsl.IRREGULAR_RATE, 'float32', 'finger_predictions')
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

    def predict(self):
        """
        Continuously checks the buffer for new data and makes predictions.
        """
        data, _ = self.exg_inlet.pull_chunk(timeout=1.0)
        
        if data is None:
            return None
    
        # EMA processing
        ema_data = self.ema_processor.process(data)
        ema_data = self.ema_processor.results_to_df(ema_data)
        ema_data = ema_data[self.ema_columns].values
        # Preprocess the data
        ema_data = self.preprocess_ema(ema_data)
        
        prediction = self.model.predict(ema_data, verbose=0)
        # ema predictions
        
        # Mean of the output
        average_output = np.mean(prediction, axis=0)
        if len(average_output) == 4:
            # Add the last finger twice
            average_output = np.append(average_output, average_output[-1])
        
        average_output = (average_output + self.prev_average_output) / 2
        self.prev_average_output = average_output
        
        binary_prediction = average_output.round().astype(int)
        
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
    model_dir = "data/s_4_26_25/models/model_0"

    # Initialize RealTimePredictor
    predictor = RealTimePredictor(
        filtered_exg_stream="filtered_exg",
        model_dir=model_dir,
        fps=5
    )

    # Run the real-time predictor
    predictor.run_loop()

if __name__ == "__main__":
    main()
