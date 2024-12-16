import numpy as np
from pylsl import StreamInlet, resolve_byprop
import pylsl
import time
import pandas as pd

class NumpyBuffer:
    def __init__(self, init_capacity, shape, dtype):
        self._dtype = dtype
        self._buffer = np.zeros((init_capacity, *shape), dtype=dtype)
        self._size = 0

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        # Handle single index
        if isinstance(idx, int):
            # Handle negative indices
            if idx < 0:
                idx += self._size
            # Check if index is out of bounds
            if idx < 0 or idx >= self._size:
                raise IndexError("Index out of bounds.")
            # Return element
            return self._buffer[idx]
        # Handle slice
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self._size)
            # Handle negative indices
            if start < 0:
                start += self._size
            if stop < 0:
                stop += self._size
            # Check if index is out of bounds
            if start < 0 or start >= self._size or stop < 0 or stop > self._size:
                raise IndexError("Index out of bounds.")
            # Return sliced buffer
            return self._buffer[start:stop:step]
        # Handle list of indices
        elif isinstance(idx, list):
            # Handle negative indices
            idx = [i if i >= 0 else i + self._size for i in idx]
            # Check if all indices are valid
            if not all(self.valid_index(i) for i in idx):
                raise IndexError("Index out of bounds.")
            # Return list of elements
            return self._buffer[idx]
        else:
            raise TypeError("Invalid index type.")

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        return self._size

    @property
    def capacity(self):
        return len(self._buffer)

    @property
    def shape(self):
        # Return shape with first dimension as size instead of capacity
        return self._buffer[:self._size].shape

    def valid_index(self, idx):
        return -self._size <= idx < self._size

    def need_to_expand(self, new_size):
        return new_size > self.capacity

    def expand(self, new_capacity=None):
        if new_capacity is None:
            new_capacity = self.capacity * 2
            
        new_buffer = np.zeros((new_capacity, *self.shape), dtype=self._dtype)
        new_buffer[:self._size] = self._buffer[:self._size]
        self._buffer = new_buffer

    def append(self, value):
        if self.need_to_expand(self._size + 1):
            new_capacity = max(self.capacity * 2, self._size + 1)
            self.expand(new_capacity)

        self._buffer[self._size] = value
        self._size += 1

    def extend(self, array: np.ndarray):
        # Check if array shape matches buffer shape
        if array.shape[1:] != self.shape[1:]:
            raise ValueError("Array shape does not match buffer shape.")
        
        array_size = array.shape[0]
        # Check if array is empty
        if array_size == 0:
            return

        new_size = array_size + self._size
        if self.need_to_expand(new_size):
            new_capacity = max(self.capacity * 2, new_size)
            self.expand(new_capacity)

        self._buffer[self._size:new_size] = array
        self._size = new_size

    def get_all(self):
        data = self._buffer[:self._size].copy()
        self._size = 0  # Clear buffer after fetching
        return data

class EMARecorder:
    def __init__(self,
                 exg_stream_name:str ="filtered_exg",
                 angle_stream_name:str ="FingerPercentages",
                 csv_filename:str ="data/exg_angle",
                 init_capacity_s:float =1024,
                 ema_spans:tuple =(1, 2, 4, 8, 16, 32, 64)
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
        self.ema_spans = ema_spans

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
        proc_flags = pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_monotonize
        self.exg_inlet = StreamInlet(exg_streams[0], max_buflen=2, processing_flags=proc_flags)
        self.angle_inlet = StreamInlet(angle_streams[0], max_buflen=2, processing_flags=proc_flags)

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

        # Initialize buffers using NumpyBuffer
        exg_dtype = np.float32
        mp_dtype = np.float32
        timestamp_dtype = np.float64
        init_capacity_exg = int(init_capacity_s * self.exg_rate)
        init_capacity_angle = int(init_capacity_s * self.angle_rate)
        
        # Buffer shapes
        timestamp_shape = ()
        exg_shape = (len(self.ema_spans), self.num_exg_channels)
        angle_shape = (self.num_angle_channels,)
        
        # Initialize buffers
        self.exg_buffer = NumpyBuffer(init_capacity=init_capacity_exg, shape=exg_shape, dtype=exg_dtype)
        self.exg_timestamps_buffer = NumpyBuffer(init_capacity=init_capacity_exg, shape=timestamp_shape, dtype=timestamp_dtype)

        self.angle_buffer = NumpyBuffer(init_capacity=init_capacity_angle, shape=angle_shape, dtype=mp_dtype)
        self.angle_timestamps_buffer = NumpyBuffer(init_capacity=init_capacity_angle, shape=timestamp_shape, dtype=timestamp_dtype)
        
        self.prev_ema_values = np.zeros(exg_shape, dtype=exg_dtype)
        self.exg_buffer_start = len(self.ema_spans) - 1 # Start saving data after the first EMA span
        
        # EMG Channel Span names
        self.exg_span_names = []
        for span in self.ema_spans:
            for i in range(self.num_exg_channels):
                self.exg_span_names.append(f"ch_{i+1}_ema_{span}")

        # Angle channel names
        self.mp_channel_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        print(f'EMG Span channel names: {self.exg_span_names}')
        print(f'Angle channel names: {self.mp_channel_names}')


    @property
    def exg_dtype(self):
        return self.exg_buffer.dtype

    @property
    def mp_dtype(self):
        return self.angle_buffer.dtype

    @property
    def timestamp_dtype(self):
        return self.exg_timestamps_buffer.dtype

    @property
    def duration_exg_s(self):
        return self.exg_buffer.size / self.exg_rate

    @property
    def duration_angle_s(self):
        return self.angle_buffer.size / self.angle_rate

    def span_to_alpha(self, span):
        return 2 / (span + 1)

    def collect_data(self):
        """Continuously collects data from the streams until interrupted."""
        # Precompute alpha values for all spans
        alphas = np.array([self.span_to_alpha(span) for span in self.ema_spans], dtype=self.exg_dtype)  # Shape: (num_spans,)
        alphas = alphas[:, np.newaxis]  # Shape: (num_spans, 1) for broadcasting
        
        print("Starting data collection. Press Ctrl+C to stop and save data.")
        try:
            while True:
                # Pull data from EMG and Angle streams
                exg_data, exg_timestamps = self.exg_inlet.pull_chunk(timeout=0.0)
                angle_data, angle_timestamps = self.angle_inlet.pull_chunk(timeout=0.0)

                if exg_data and exg_timestamps:
                    # exg_data shape: (N, num_exg_channels)
                    exg_data = np.array(exg_data, dtype=self.exg_dtype)
                    exg_data = np.abs(exg_data, out=exg_data)
                    
                    # Initialize array to store EMA
                    num_samples = len(exg_data)
                    new_exg_samples = np.zeros((num_samples, len(self.ema_spans), self.num_exg_channels), dtype=self.exg_dtype)

                    for i in range(num_samples):
                        sample = exg_data[i]  # Shape: (num_channels,)
                        # Update EMA: alpha * sample + (1 - alpha) * prev_ema
                        # Vectorized Equivalent
                        # self.prev_ema_values = alphas * sample + (1 - alphas) * self.prev_ema_values
                        # Inplace Equivalent
                        self.prev_ema_values *= 1 - alphas
                        self.prev_ema_values += alphas * sample
                        new_exg_samples[i] = self.prev_ema_values

                    # Append the processed exg data and timestamps
                    self.exg_buffer.extend(new_exg_samples)
                    self.exg_timestamps_buffer.extend(np.array(exg_timestamps, dtype=self.timestamp_dtype))

                if angle_data and angle_timestamps:
                    angle_data = np.array(angle_data, dtype=self.mp_dtype)
                    # Append angle data and timestamps
                    self.angle_buffer.extend(angle_data)
                    self.angle_timestamps_buffer.extend(np.array(angle_timestamps, dtype=self.timestamp_dtype))

                # Sleep briefly to prevent CPU overuse
                time_to_sleep = self.update_time - 0.001
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Preparing to save data...")
            self.save_data()

    def save_data(self):
        """Synchronizes all buffered data and writes to the CSV file."""
        # Get EXG data
        if self.angle_buffer.size == 0 or self.exg_buffer.size <= self.exg_buffer_start:
            print("No data collected. Exiting without saving.")
            return
        
        exg_timestamps = self.exg_timestamps_buffer.get_all()[self.exg_buffer_start:]  # shape: (N,)
        exg_samples = self.exg_buffer.get_all()[self.exg_buffer_start:]  # shape: (N, len(ema_spans), num_exg_channels)
        self.exg_buffer_start = 0

        # Get Angle data
        angle_timestamps = self.angle_timestamps_buffer.get_all()  # shape: (M,)
        angle_samples = self.angle_buffer.get_all()  # shape: (M, num_angle_channels)

        # Synchronize start and end times
        start = max(exg_timestamps[0], angle_timestamps[0])
        end = min(exg_timestamps[-1], angle_timestamps[-1])

        exg_mask = (exg_timestamps >= start) & (exg_timestamps <= end)
        angle_mask = (angle_timestamps >= start) & (angle_timestamps <= end)

        exg_timestamps = exg_timestamps[exg_mask]
        exg_samples = exg_samples[exg_mask]

        angle_timestamps = angle_timestamps[angle_mask]
        angle_samples = angle_samples[angle_mask]

        # (N, len(ema_spans), num_exg_channels) -> (N, len(ema_spans)*num_exg_channels)
        exg_samples_reshaped = exg_samples.reshape(-1, len(self.ema_spans)*self.num_exg_channels)

        # Create DataFrames
        exg_df = pd.DataFrame(exg_samples_reshaped, index=exg_timestamps, columns=self.exg_span_names)
        angle_df = pd.DataFrame(angle_samples, index=angle_timestamps, columns=self.mp_channel_names)

        # Set timestamps as timedelta
        exg_df.index = pd.to_timedelta(exg_df.index, unit='s')
        angle_df.index = pd.to_timedelta(angle_df.index, unit='s')

        # Merge as-of nearest timestamp
        tolerance = pd.to_timedelta(self.update_time, unit='s')
        merged_df = pd.merge_asof(angle_df, exg_df, left_index=True, right_index=True, direction='nearest', tolerance=tolerance)

        # Convert timestamps to milliseconds
        merged_df.index = merged_df.index.astype(int) // 10**6
        # Subtract the first timestamp to start from 0
        merged_df.index -= merged_df.index[0]
        # Name index as 'timestamp'
        merged_df.index.name = 'timestamp'

        # Save to CSV
        file_name = f"{self.csv_filename}_{int(time.time())}.csv"
        merged_df.to_csv(file_name)

        print(f"Data saved to '{file_name}'. Exiting...")


def main():
    recorder = EMARecorder(
        exg_stream_name="filtered_exg",
        angle_stream_name="FingerPercentages",
        csv_filename="data/s_1/exg_angle",
        ema_spans=[1, 2, 4, 8, 16, 32, 64]  # example spans
    )
    recorder.collect_data()


if __name__ == "__main__":
    main()