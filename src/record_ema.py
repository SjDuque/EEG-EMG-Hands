# record_ema.py
import numpy as np
from pylsl import StreamInlet, resolve_byprop
import pylsl  
import time
import pandas as pd
import os

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
    """
    _summary_: Class to record EXG and Angle data from LSL streams and save to CSV files.
    Applies exponential moving average to the EXG data.
    """
    def __init__(self,
                 exg_stream_name:str ="filtered_exg",
                 angle_stream_name:str ="FingerPercentages",
                 csv_dir:str ="data/session",
                 save_exg:bool =True,
                 save_angle:bool =True,
                 save_merged:bool =True,
                 init_capacity_s:float =1024,
                 ema_spans:tuple =(1, 2, 4, 8, 16, 32, 64)
                 ):
        """
        Initializes the recorder.

        :param exg_stream_name: Name of the EMG stream.
        :param angle_stream_name: Name of the Angle stream.
        :param csv_dir: Output CSV files directory.
        :param ema_spans: Exponential moving average spans.
        """
        self.exg_stream_name = exg_stream_name
        self.angle_stream_name = angle_stream_name
        self.csv_dir = csv_dir
        self.ema_spans = ema_spans
        
        self.save_exg = save_exg
        self.save_angle = save_angle
        self.save_merged = save_merged
        
        self.start_time = int(pylsl.local_clock() * 1000)

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
        # print('Angle Channels', angle_info.desc().child('channels').as_string().split(','))
        self.angle_rate = angle_info.nominal_srate()
        if self.angle_rate <= 0:
            raise ValueError("Angle stream sampling rate is not set or zero.")
        self.num_angle_channels = angle_info.channel_count()
        if self.num_angle_channels <= 0:
            raise ValueError("Angle stream channel_count is not set or zero.")

        self.update_time = min(1/self.exg_rate, 1/self.angle_rate)

        # Initialize buffers using NumpyBuffer
        ema_dtype = np.float32
        mp_dtype = np.float32
        timestamp_dtype = np.float64
        init_capacity_ema = int(init_capacity_s * self.exg_rate)
        init_capacity_angle = int(init_capacity_s * self.angle_rate)
        
        # Buffer shapes
        timestamp_shape = ()
        ema_shape = (self.num_exg_channels, len(self.ema_spans) + 1)
        angle_shape = (self.num_angle_channels,)
        
        # Initialize buffers
        self.ema_buffer = NumpyBuffer(init_capacity=init_capacity_ema, shape=ema_shape, dtype=ema_dtype)
        self.ema_timestamp_buffer = NumpyBuffer(init_capacity=init_capacity_ema, shape=timestamp_shape, dtype=timestamp_dtype)

        self.angle_buffer = NumpyBuffer(init_capacity=init_capacity_angle, shape=angle_shape, dtype=mp_dtype)
        self.angle_timestamp_buffer = NumpyBuffer(init_capacity=init_capacity_angle, shape=timestamp_shape, dtype=timestamp_dtype)
        
        self.alphas = np.array([self.span_to_alpha(span) for span in self.ema_spans], dtype=self.ema_dtype)  # Shape: (num_spans,)
        self.alphas = self.alphas[np.newaxis, :]
        
        # Initialize EMA values
        
        prev_ema_shape = (self.num_exg_channels, len(self.ema_spans))
        self.prev_ema_values = np.zeros(prev_ema_shape, dtype=ema_dtype)
        
        self.ema_buffer_start = len(self.ema_spans) - 1 # Start saving data after the first EMA span
        
        # Channel -> Span
        self.ema_span_names = [f"ch_{i+1}_ema_{span}" for i in range(self.num_exg_channels) for span in self.ema_spans]
        self.exg_span_names = []
        
        for i in range(self.num_exg_channels):
            self.exg_span_names.append(f"ch_{i+1}_raw")
            for span in self.ema_spans:
                self.exg_span_names.append(f"ch_{i+1}_ema_{span}")

        # Angle channel names
        self.mp_channel_names = self._get_angle_channel_names(angle_info)

        print(f'EMG Span channel names: {self.exg_span_names}')
        print(f'Angle channel names: {self.mp_channel_names}')
        
    def _get_angle_channel_names(self, angle_info):
        """
        Extracts the angle channel names from the LSL stream's metadata.

        :param angle_info: pylsl.StreamInfo object for the angle stream.
        :return: List of channel names.
        """
        angle_desc = angle_info.desc()
        child_element = angle_desc.child('channels').child('channel')
        
        channel_names = []
        while not child_element.empty():
            channel_names.append(child_element.child_value('label'))
            child_element = child_element.next_sibling()

        if not channel_names:
            # Fallback to default names if labels are not provided
            channel_names = [f"angle_{i+1}" for i in range(self.num_angle_channels)]
            print("No channel labels found in angle stream metadata. Using default channel names.")
        else:
            print(f"Retrieved angle channel names: {channel_names}")

        return channel_names

    @property
    def ema_dtype(self):
        return self.ema_buffer.dtype

    @property
    def mp_dtype(self):
        return self.angle_buffer.dtype

    @property
    def timestamp_dtype(self):
        return self.ema_timestamp_buffer.dtype

    @property
    def duration_exg_s(self):
        return self.ema_buffer.size / self.exg_rate

    @property
    def duration_angle_s(self):
        return self.angle_buffer.size / self.angle_rate

    def span_to_alpha(self, span):
        return 2 / (span + 1)
    
    def update(self):
        # Pull data from EMG and Angle streams
        exg_data, ema_timestamps = self.exg_inlet.pull_chunk(timeout=0.0)
        angle_data, angle_timestamps = self.angle_inlet.pull_chunk(timeout=0.0)

        if exg_data and ema_timestamps:
            # exg_data shape: (N, num_exg_channels)
            raw_exg_data = np.array(exg_data, dtype=self.ema_dtype)
            rect_exg_data = np.abs(raw_exg_data)
            
            # Initialize array to store EMA
            num_samples = len(rect_exg_data)
            new_ema_samples = np.zeros((num_samples, *self.ema_buffer.shape[1:]), dtype=self.ema_dtype)
            new_ema_samples[:, :, 0] = raw_exg_data  # Save raw data in the first column

            for i in range(num_samples):
                sample = rect_exg_data[i]  # Shape: (num_channels,)
                # Update EMA: alpha * sample + (1 - alpha) * prev_ema
                # self.prev_ema_values = alphas * sample + (1 - alphas) * self.prev_ema_values
                # Inplace Equivalent
                self.prev_ema_values *= (1 - self.alphas)  # Shape: (num_channels, len_spans)
                self.prev_ema_values += self.alphas * sample[:, np.newaxis]  # Broadcasting alpha
                new_ema_samples[i, :, 1:] = self.prev_ema_values

            # Append the processed exg data and timestamps
            self.ema_buffer.extend(new_ema_samples)
            self.ema_timestamp_buffer.extend(np.array(ema_timestamps, dtype=self.timestamp_dtype))

        if angle_data and angle_timestamps:
            angle_data = np.array(angle_data, dtype=self.mp_dtype)
            # Append angle data and timestamps
            self.angle_buffer.extend(angle_data)
            self.angle_timestamp_buffer.extend(np.array(angle_timestamps, dtype=self.timestamp_dtype))

    def collect_and_save_loop(self):
        """Continuously collects data from the streams until interrupted."""
        # Precompute alpha values for all spans
        alphas = np.array([self.span_to_alpha(span) for span in self.ema_spans], dtype=self.ema_dtype)  # Shape: (num_spans,)
        alphas = alphas[np.newaxis, :] # Shape: (1, num_spans)
        
        print("Starting data collection loop. Press Ctrl+C to stop and save data.")
        try:
            while True:
                self.update()
                # Sleep briefly to prevent CPU overuse
                time_to_sleep = self.update_time - 0.001
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Preparing to save data...")
            self.save_data()
            
    def get_data(self):
        """
        Synchronizes the buffered data and returns as DataFrames,
        Returns: Tuple of DataFrames (ema_df, angle_df, merged_df)
        """
        # Get EXG data
        if self.angle_buffer.size == 0 or self.ema_buffer.size <= self.ema_buffer_start:
            return None
        
        ema_timestamps = self.ema_timestamp_buffer.get_all()[self.ema_buffer_start:]  # shape: (N,)
        ema_samples = self.ema_buffer.get_all()[self.ema_buffer_start:]  # shape: (N, len(ema_spans), num_exg_channels)
        self.ema_buffer_start = 0

        # Get Angle data
        angle_timestamps = self.angle_timestamp_buffer.get_all()  # shape: (M,)
        angle_samples = self.angle_buffer.get_all()  # shape: (M, num_angle_channels)

        # Synchronize start and end times by taking the maximum start and minimum end
        # Then filter out the data that is outside the synchronized range
        start = max(ema_timestamps[0], angle_timestamps[0])
        end = min(ema_timestamps[-1], angle_timestamps[-1])
        
        exg_mask = (ema_timestamps >= start) & (ema_timestamps <= end)
        angle_mask = (angle_timestamps >= start) & (angle_timestamps <= end)

        ema_timestamps = ema_timestamps[exg_mask]
        ema_samples = ema_samples[exg_mask]
        angle_timestamps = angle_timestamps[angle_mask]
        angle_samples = angle_samples[angle_mask]
        
        # (N, num_exg_channels, len(ema_spans)) -> (N, self.num_exg_channels * len(self.ema_spans))
        ema_shape = ema_samples.shape
        ema_samples_reshaped = ema_samples.reshape(-1, ema_shape[-1] * ema_shape[-2])

        # Create DataFrames
        ema_df = pd.DataFrame(ema_samples_reshaped, index=ema_timestamps, columns=self.exg_span_names)
        angle_df = pd.DataFrame(angle_samples, index=angle_timestamps, columns=self.mp_channel_names)

        # Set timestamps as timedelta
        ema_df.index = pd.to_timedelta(ema_df.index, unit='s')
        angle_df.index = pd.to_timedelta(angle_df.index, unit='s')

        # Merge as-of nearest timestamp
        tolerance = pd.to_timedelta(self.update_time, unit='s')
        merged_df = pd.merge_asof(angle_df, ema_df, left_index=True, right_index=True, direction='nearest', tolerance=tolerance)
        
        # Convert time index to milliseconds formatted as integers
        ema_df.index = (ema_df.index.total_seconds() * 1000).round().astype(int)
        angle_df.index = (angle_df.index.total_seconds() * 1000).round().astype(int)
        merged_df.index = (merged_df.index.total_seconds() * 1000).round().astype(int)
        
        # Set index name
        ema_df.index.name = 'timestamp'
        angle_df.index.name = 'timestamp'
        merged_df.index.name = 'timestamp'
        
        # Set start time to 0
        self.start_time = min(ema_df.index[0], angle_df.index[0], merged_df.index[0], self.start_time)
        ema_df.index -= self.start_time
        angle_df.index -= self.start_time
        merged_df.index -= self.start_time
        
        return ema_df, angle_df, merged_df

    def save_data(self):
        """Synchronizes all buffered data and writes to the CSV file."""
        ema_df, angle_df, merged_df = self.get_data()
        
        # Create CSV directory if it does not exist
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)

        # Save to CSV
        timestamp = int(time.time())
        
        # Save dataframes to CSV files
        if self.save_exg:
            if not os.path.exists(f"{self.csv_dir}/exg"):
                os.makedirs(f"{self.csv_dir}/exg")
            ema_filename = f"{self.csv_dir}/exg/data_{timestamp}.csv"
            ema_df.to_csv(ema_filename)
            print(f"EMA Dataframe saved to '{ema_filename}'.")
        
        if self.save_angle:
            if not os.path.exists(f"{self.csv_dir}/angle"):
                os.makedirs(f"{self.csv_dir}/angle")
            angle_filename = f"{self.csv_dir}/angle/data_{timestamp}.csv"
            angle_df.to_csv(angle_filename)
            print(f"Angle Dataframe saved to '{angle_filename}'.")
        
        if self.save_merged:
            merged_filename = f"{self.csv_dir}/data_{timestamp}.csv"
            merged_df.to_csv(merged_filename)
            print(f"Merged Dataframe saved to '{merged_filename}'.")


def main():
    recorder = EMARecorder(
        exg_stream_name="filtered_exg",
        angle_stream_name="FingerPercentages",
        csv_dir="data/s_4",
        ema_spans=[1, 2, 4, 8, 16, 32, 64],
        
    )
    recorder.collect_and_save_loop()


if __name__ == "__main__":
    main()