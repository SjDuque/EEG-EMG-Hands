import numpy as np
from pylsl import StreamInlet, resolve_byprop, proc_ALL
import time
import pandas as pd

class EMARecorder:
    def __init__(self, 
                 emg_stream_name="filtered_exg", 
                 angle_stream_name="FingerPercentages",
                 csv_filename="data/emg_mp",
                 ema_spans=(1, 2, 4, 8, 16, 32, 64)
                 ):
        """
        Initializes the recorder.

        :param emg_stream_name: Name of the EMG stream.
        :param angle_stream_name: Name of the Angle stream.
        :param csv_filename: Output CSV file name.
        :param ema_spans: Exponential moving average spans.
        """
        self.emg_stream_name = emg_stream_name
        self.angle_stream_name = angle_stream_name
        self.csv_filename = csv_filename

        # Resolve streams
        print("Resolving EMG (EXG) stream...")
        emg_streams = resolve_byprop('name', self.emg_stream_name, timeout=5)
        if not emg_streams:
            raise RuntimeError(f"No EMG stream found with name '{self.emg_stream_name}'.")

        print("Resolving angle (MP) stream...")
        angle_streams = resolve_byprop('name', self.angle_stream_name, timeout=5)
        if not angle_streams:
            raise RuntimeError(f"No angle stream found with name '{self.angle_stream_name}'.")

        # Create inlets
        self.emg_inlet = StreamInlet(emg_streams[0], max_buflen=2, processing_flags=proc_ALL)
        self.angle_inlet = StreamInlet(angle_streams[0], max_buflen=2, processing_flags=proc_ALL)

        # Get stream info
        emg_info = self.emg_inlet.info()
        self.emg_rate = emg_info.nominal_srate()
        if self.emg_rate <= 0:
            raise ValueError("EMG stream sampling rate is not set or zero.")
        self.num_emg_channels = emg_info.channel_count()
        if self.num_emg_channels <= 0:
            raise ValueError("EMG stream channel_count is not set or zero.")

        angle_info = self.angle_inlet.info()
        print('Angle Channels', angle_info.desc().child('channels').as_string().split(','))
        self.angle_rate = angle_info.nominal_srate()
        if self.angle_rate <= 0:
            raise ValueError("Angle stream sampling rate is not set or zero.")
        self.num_angle_channels = angle_info.channel_count()
        if self.num_angle_channels <= 0:
            raise ValueError("Angle stream channel_count is not set or zero.")
        
        self.update_time = min(1/self.emg_rate, 1/self.angle_rate)

        # Initialize buffers
        self.dtype = np.float32
        self.emg_buffer = {span: [] for span in ema_spans}
        self.emg_buffer['timestamp'] = []
        self.ema_spans = ema_spans
        self.angle_buffer = {'timestamp': [], 'angles': []}
        
        # Extract EMG Channel Span names
        self.emg_span_names = [f"ch_{i+1}_ema_{span}" for span in self.ema_spans for i in range(self.num_emg_channels)]
        # Extract Angle channel names
        self.mp_channel_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            
        print(f'EMG Span channel names: {self.emg_span_names}')
        print(f'Angle channel names: {self.mp_channel_names}')
        
        # Initialize previous EMA values
        initial_emg_sample, initial_emg_timestamp = self.emg_inlet.pull_sample()
        self.prev_ema_values = {span: np.array(initial_emg_sample) for span in self.ema_spans}
        
        # Add initial sample to buffer
        for span in self.ema_spans:
            self.emg_buffer[span].append(initial_emg_sample)
        self.emg_buffer['timestamp'].append(initial_emg_timestamp)
        
    def span_to_alpha(self, span):
        return 2 / (span + 1)

    def collect_data(self):
        """Continuously collects data from the streams until interrupted."""
        print("Starting data collection. Press Ctrl+C to stop and save data.")
        try:
            while True:
                # Pull data from EMG and Angle streams
                emg_data, emg_timestamps = self.emg_inlet.pull_chunk(timeout=0.0)
                angle_data, angle_timestamps = self.angle_inlet.pull_chunk(timeout=0.0)
                
                if emg_data:
                    # Rectify and apply exponential moving average to EMG data
                    new_emg_buffer = {}
                    for span in self.ema_spans:
                        # Initialize buffer for each span
                        new_emg_buffer[span] = []
                        alpha = self.span_to_alpha(span)
                        # Apply EMA to each sample
                        for sample in emg_data:
                            sample = np.array(sample).abs()
                            self.prev_ema_values[span] = alpha * sample + (1 - alpha) * self.prev_ema_values[span]
                            new_emg_buffer[span].append(self.prev_ema_values[span])
                            
                    # Timestamps are the same for all spans
                    self.emg_buffer['timestamp'].extend(emg_timestamps)
                    for span in self.ema_spans:
                        self.emg_buffer[span].extend(new_emg_buffer[span])

                if angle_data:
                    self.angle_buffer['timestamp'].extend(angle_timestamps)
                    self.angle_buffer['angles'].extend(angle_data)

                # Sleep briefly to prevent CPU overuse
                time.sleep(self.update_time-0.001)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Preparing to save data...")
            self.save_data()

    def save_data(self):
        """Synchronizes all buffered data and writes to the CSV file."""
        emg_timestamps = np.array(self.emg_buffer['timestamp'], dtype=np.float64)
        angle_timestamps = np.array(self.angle_buffer['timestamp'], dtype=np.float64)

        if emg_timestamps.size == 0 or angle_timestamps.size == 0:
            print("No data collected. Exiting without saving.")
            return
        
        emg_samples = []
        for span in self.ema_spans:
            emg_samples.append(np.array(self.emg_buffer[span], dtype=self.dtype))
        
        # Stack all EMG samples based on timestamp
        emg_samples = np.stack(emg_samples, axis=-1)
        angle_samples = np.array(self.angle_buffer['angles'], dtype=self.dtype)
        
        start = max(emg_timestamps[0], angle_timestamps[0])
        end = min(emg_timestamps[-1], angle_timestamps[-1])
        
        emg_mask = (emg_timestamps >= start) & (emg_timestamps <= end)
        angle_mask = (angle_timestamps >= start) & (angle_timestamps <= end)
        
        emg_timestamps = emg_timestamps[emg_mask]
        emg_samples = emg_samples[emg_mask]
        angle_timestamps = angle_timestamps[angle_mask]
        angle_samples = angle_samples[angle_mask]
        # Synchronize data based on nearest timestamps using Pandas
        
        emg_df = pd.DataFrame(emg_samples, index=emg_timestamps, columns=self.emg_span_names)
        angle_df = pd.DataFrame(angle_samples, index=angle_timestamps, columns=self.mp_channel_names)
        # Set timestamps as timedelta type
        emg_df.index = pd.to_timedelta(emg_df.index, unit='s')
        angle_df.index = pd.to_timedelta(angle_df.index, unit='s')
        
        # Merge as of the nearest timestamp based
        tolerance = self.update_time
        tolerance = pd.to_timedelta(tolerance, unit='s')
        merged_df = pd.merge_asof(angle_df, emg_df, left_index=True, right_index=True, direction='nearest', tolerance=tolerance)
        
        # Convert timestamps to milliseconds
        merged_df.index = merged_df.index.astype(int) // 10**6
        
        # Subtract the first timestamp to start from 0
        merged_df.index -= merged_df.index[0]
        
        # Name index as 'time' and reset index
        merged_df.index.name = 'timestamp'
        
        # Save to CSV using timestamp
        file_name = f"{self.csv_filename}_{int(time.time())}.csv"
        merged_df.to_csv(f"{file_name}")
        
        print(f"Data saved to '{file_name}'. Exiting...")
        
def main():
    recorder = EMARecorder(
        emg_stream_name="filtered_exg",
        angle_stream_name="FingerPercentages",
        csv_filename="data/s_0/emg_mp",
    )
    recorder.collect_data()

if __name__ == "__main__":
    main()