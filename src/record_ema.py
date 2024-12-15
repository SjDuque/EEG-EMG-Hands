import numpy as np
from pylsl import StreamInlet, resolve_byprop, proc_ALL
import time
import pandas as pd

class NumpyBuffer:
    def __init__(self, init_capacity, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.buffer = np.zeros((init_capacity, *shape), dtype=dtype)
        self.size = 0

    def __len__(self):
        return self.size
    
    def capacity(self):
        return len(self.buffer)
    
    def valid_index(self, idx):
        return -self.size <= idx < self.size
    
    def __setitem__(self, idx, value):
        self[idx] = value

    def __getitem__(self, idx):
        # Handle single index
        if isinstance(idx, int):
            # Handle negative indices
            if idx < 0:
                idx += self.size
            # Check if index is out of bounds
            if idx < 0 or idx >= self.size:
                raise IndexError("Index out of bounds.")
            # Return element
            return self.buffer[idx]
        # Handle slice
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.size)
            # Handle negative indices
            if start < 0:
                start += self.size
            if stop < 0:
                stop += self.size    
            # Check if index is out of bounds
            if start < 0 or start >= self.size or stop < 0 or stop > self.size:
                raise IndexError("Index out of bounds.")
            # Return sliced buffer
            return self.buffer[start:stop:step]
        # Handle list of indices
        elif isinstance(idx, list):
            # Handle negative indices
            idx = [i if i >= 0 else i + self.size for i in idx]
            # Check if all indices are valid
            if not all(self.valid_index(i) for i in idx):
                raise IndexError("Index out of bounds.")
            # Return list of elements
            return self.buffer[idx]
        else:
            raise TypeError("Invalid index type.")
        
    def need_to_expand(self, new_size):
        return new_size > self.capacity()

    def expand(self, new_capacity=None):
        if new_capacity is None:
            new_capacity = self.capacity() * 2
        new_buffer = np.zeros((new_capacity, *self.shape), dtype=self.dtype)
        new_buffer[:self.size] = self.buffer[:self.size]
        self.buffer = new_buffer

    def extend(self, array: np.ndarray):
        new_size = len(array) + self.size
        if self.need_to_expand(new_size):
            new_capacity = max(self.capacity() * 2, new_size)
            self.expand(new_capacity)
        self.buffer[self.size:new_size] = array

        self.size += new_size

    def get_all(self):
        data = self.buffer[:self.size].copy()
        self.size = 0  # Clear buffer after fetching
        return data

class EMARecorder:
    def __init__(self, 
                 exg_stream_name="filtered_exg", 
                 angle_stream_name="FingerPercentages",
                 csv_filename="data/exg_mp",
                 ema_spans=(1, 2, 4, 8, 16, 32, 64)
                 ):
        """
        Initializes the recorder.

        :param exg_stream_name: Name of the EMG stream.
        :param angle_stream_name: Name of the Angle stream.
        :param csv_filename: Output CSV file name.
        :param ema_spans: Exponential moving average spans.
        """
        self.exg_stream_name = exg_stream_name
        self.angle_stream_name = angle_stream_name
        self.csv_filename = csv_filename

        # Resolve streams
        print("Resolving EMG (EXG) stream...")
        exg_streams = resolve_byprop('name', self.exg_stream_name, timeout=5)
        if not exg_streams:
            raise RuntimeError(f"No EMG stream found with name '{self.exg_stream_name}'.")

        print("Resolving angle (MP) stream...")
        angle_streams = resolve_byprop('name', self.angle_stream_name, timeout=5)
        if not angle_streams:
            raise RuntimeError(f"No angle stream found with name '{self.angle_stream_name}'.")

        # Create inlets
        self.exg_inlet = StreamInlet(exg_streams[0], max_buflen=2, processing_flags=proc_ALL)
        self.angle_inlet = StreamInlet(angle_streams[0], max_buflen=2, processing_flags=proc_ALL)

        # Get stream info
        exg_info = self.exg_inlet.info()
        self.exg_rate = exg_info.nominal_srate()
        if self.exg_rate <= 0:
            raise ValueError("EMG stream sampling rate is not set or zero.")
        self.num_exg_channels = exg_info.channel_count()
        if self.num_exg_channels <= 0:
            raise ValueError("EMG stream channel_count is not set or zero.")

        angle_info = self.angle_inlet.info()
        print('Angle Channels', angle_info.desc().child('channels').as_string().split(','))
        self.angle_rate = angle_info.nominal_srate()
        if self.angle_rate <= 0:
            raise ValueError("Angle stream sampling rate is not set or zero.")
        self.num_angle_channels = angle_info.channel_count()
        if self.num_angle_channels <= 0:
            raise ValueError("Angle stream channel_count is not set or zero.")
        
        self.update_time = min(1/self.exg_rate, 1/self.angle_rate)

        # Initialize buffers
        self.dtype = np.float32
        self.exg_buffer = {span: [] for span in ema_spans}
        self.exg_buffer['timestamp'] = []
        self.ema_spans = ema_spans
        self.angle_buffer = {'timestamp': [], 'angles': []}
        
        # Extract EMG Channel Span names
        self.exg_span_names = [f"ch_{i+1}_ema_{span}" for span in self.ema_spans for i in range(self.num_exg_channels)]
        # Extract Angle channel names
        self.mp_channel_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            
        print(f'EMG Span channel names: {self.exg_span_names}')
        print(f'Angle channel names: {self.mp_channel_names}')
        
        # Initialize previous EMA values
        initial_exg_sample, initial_exg_timestamp = self.exg_inlet.pull_sample()
        self.prev_ema_values = {span: np.array(initial_exg_sample) for span in self.ema_spans}
        
        # Add initial sample to buffer
        for span in self.ema_spans:
            self.exg_buffer[span].append(initial_exg_sample)
        self.exg_buffer['timestamp'].append(initial_exg_timestamp)
        
    def span_to_alpha(self, span):
        return 2 / (span + 1)

    def collect_data(self):
        """Continuously collects data from the streams until interrupted."""
        print("Starting data collection. Press Ctrl+C to stop and save data.")
        try:
            while True:
                # Pull data from EMG and Angle streams
                exg_data, exg_timestamps = self.exg_inlet.pull_chunk(timeout=0.0)
                angle_data, angle_timestamps = self.angle_inlet.pull_chunk(timeout=0.0)
                
                if exg_data:
                    # Rectify and apply exponential moving average to EMG data
                    new_exg_buffer = {}
                    for span in self.ema_spans:
                        # Initialize buffer for each span
                        new_exg_buffer[span] = []
                        alpha = self.span_to_alpha(span)
                        # Apply EMA to each sample
                        for sample in exg_data:
                            sample = np.array(sample).abs()
                            self.prev_ema_values[span] = alpha * sample + (1 - alpha) * self.prev_ema_values[span]
                            new_exg_buffer[span].append(self.prev_ema_values[span])
                            
                    # Timestamps are the same for all spans
                    self.exg_buffer['timestamp'].extend(exg_timestamps)
                    for span in self.ema_spans:
                        self.exg_buffer[span].extend(new_exg_buffer[span])

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
        exg_timestamps = np.array(self.exg_buffer['timestamp'], dtype=np.float64)
        angle_timestamps = np.array(self.angle_buffer['timestamp'], dtype=np.float64)

        if exg_timestamps.size == 0 or angle_timestamps.size == 0:
            print("No data collected. Exiting without saving.")
            return
        
        exg_samples = []
        for span in self.ema_spans:
            exg_samples.append(np.array(self.exg_buffer[span], dtype=self.dtype))
        
        # Stack all EMG samples based on timestamp
        exg_samples = np.stack(exg_samples, axis=-1)
        angle_samples = np.array(self.angle_buffer['angles'], dtype=self.dtype)
        
        start = max(exg_timestamps[0], angle_timestamps[0])
        end = min(exg_timestamps[-1], angle_timestamps[-1])
        
        exg_mask = (exg_timestamps >= start) & (exg_timestamps <= end)
        angle_mask = (angle_timestamps >= start) & (angle_timestamps <= end)
        
        exg_timestamps = exg_timestamps[exg_mask]
        exg_samples = exg_samples[exg_mask]
        angle_timestamps = angle_timestamps[angle_mask]
        angle_samples = angle_samples[angle_mask]
        # Synchronize data based on nearest timestamps using Pandas
        
        exg_df = pd.DataFrame(exg_samples, index=exg_timestamps, columns=self.exg_span_names)
        angle_df = pd.DataFrame(angle_samples, index=angle_timestamps, columns=self.mp_channel_names)
        # Set timestamps as timedelta type
        exg_df.index = pd.to_timedelta(exg_df.index, unit='s')
        angle_df.index = pd.to_timedelta(angle_df.index, unit='s')
        
        # Merge as of the nearest timestamp based
        tolerance = self.update_time
        tolerance = pd.to_timedelta(tolerance, unit='s')
        merged_df = pd.merge_asof(angle_df, exg_df, left_index=True, right_index=True, direction='nearest', tolerance=tolerance)
        
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
        exg_stream_name="filtered_exg",
        angle_stream_name="FingerPercentages",
        csv_filename="data/s_0/exg_mp",
    )
    recorder.collect_data()

if __name__ == "__main__":
    main()