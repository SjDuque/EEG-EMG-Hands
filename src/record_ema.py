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
                 status_stream_name:str ="FingerStatus",
                 csv_dir:str ="data/session",
                 save_exg:bool =True,
                 save_angle:bool =True,
                 save_status:bool =True,
                 save_merged:bool =True,
                 init_capacity_s:float =1024,
                 update_interval:float =0.01,
                 ema_spans:tuple =(1, 2, 4, 8, 16, 32, 64)
                 ):
        """
        Initializes the recorder.

        :param exg_stream_name: Name of the EMG stream.
        :param angle_stream_name: Name of the Angle stream.
        :param csv_dir: Output CSV files directory.
        :param ema_spans: Exponential moving average spans.
        """
        self.csv_dir = csv_dir
        self.ema_spans = ema_spans
        
        self.save_exg = save_exg
        self.save_angle = save_angle
        self.save_merged = save_merged
        self.save_status = save_status
    
        self.update_interval = update_interval
        
        # Processing Flags
        proc_flags = pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_monotonize

        # EXG Stream
        self.exg_stream_name = exg_stream_name
        print("Resolving EMG (EXG) stream...")
        exg_streams = resolve_byprop('name', self.exg_stream_name, timeout=5)
        if not exg_streams:
            raise RuntimeError(f"No EMG stream found with name '{self.exg_stream_name}'.")

        self.exg_inlet = StreamInlet(exg_streams[0], max_buflen=512, processing_flags=proc_flags)
        exg_info = self.exg_inlet.info()
        self.exg_rate = exg_info.nominal_srate()
        if self.exg_rate <= 0:
            raise ValueError("EMG stream sampling rate is not set or zero.")
        self.num_exg_channels = exg_info.channel_count()
        if self.num_exg_channels <= 0:
            raise ValueError("EMG stream channel_count is not set or zero.")

        # Angle Stream
        self.angle_stream_name = angle_stream_name
        print("Resolving angle (MP) stream...")
        angle_streams = resolve_byprop('name', self.angle_stream_name, timeout=5)
        if not angle_streams:
            raise RuntimeError(f"No angle stream found with name '{self.angle_stream_name}'.")
        
        self.angle_inlet = StreamInlet(angle_streams[0], max_buflen=512, processing_flags=proc_flags)
        angle_info = self.angle_inlet.info()
        self.angle_rate = angle_info.nominal_srate()
        if self.angle_rate <= 0:
            raise ValueError("Angle stream sampling rate is not set or zero.")
        self.num_angle_channels = angle_info.channel_count()
        if self.num_angle_channels <= 0:
            raise ValueError("Angle stream channel_count is not set or zero.")
        
        self.mp_channel_names = self._get_angle_channel_names(angle_info)
        
        # Status Stream
        self.status_stream_name = status_stream_name
        print("Resolving status stream...")
        status_streams = resolve_byprop('name', self.status_stream_name, timeout=5)
        self.status_inlet = StreamInlet(status_streams[0], max_buflen=512, processing_flags=proc_flags)
        status_info = self.status_inlet.info()
        self.status_rate = status_info.nominal_srate()
        if self.status_rate <= 0:
            raise ValueError("Status stream sampling rate is not set or zero.")
        self.num_status_channels = status_info.channel_count()
        if self.num_status_channels <= 0:
            raise ValueError("Status stream channel_count is not set or zero.")
        
        self.status_channel_names = self._get_status_channel_names(status_info)

        # Initialize buffers using NumpyBuffer
        ema_dtype = np.float32
        mp_dtype = np.float32
        timestamp_dtype = np.float64
        init_capacity_ema = int(init_capacity_s * self.exg_rate)
        init_capacity_angle = int(init_capacity_s * self.angle_rate)
        init_capacity_status = int(init_capacity_s * self.status_rate)
        
        # Buffer shapes
        timestamp_shape = ()
        ema_shape = (self.num_exg_channels, len(self.ema_spans) + 1)
        angle_shape = (self.num_angle_channels,)
        status_shape = (self.num_status_channels,)
        
        # Initialize buffers
        self.ema_buffer = NumpyBuffer(init_capacity=init_capacity_ema, shape=ema_shape, dtype=ema_dtype)
        self.ema_timestamp_buffer = NumpyBuffer(init_capacity=init_capacity_ema, shape=timestamp_shape, dtype=timestamp_dtype)

        self.angle_buffer = NumpyBuffer(init_capacity=init_capacity_angle, shape=angle_shape, dtype=mp_dtype)
        self.angle_timestamp_buffer = NumpyBuffer(init_capacity=init_capacity_angle, shape=timestamp_shape, dtype=timestamp_dtype)
        
        self.status_buffer = NumpyBuffer(init_capacity=init_capacity_status, shape=status_shape, dtype=mp_dtype)
        self.status_timestamp_buffer = NumpyBuffer(init_capacity=init_capacity_status, shape=timestamp_shape, dtype=timestamp_dtype)
        
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
    
    def _get_status_channel_names(self, status_info):
        """
        Extracts the status channel names from the LSL stream's metadata.

        :param status_info: pylsl.StreamInfo object for the status stream.
        :return: List of channel names.
        """
        status_desc = status_info.desc()
        child_element = status_desc.child('channels').child('channel')
        
        channel_names = []
        while not child_element.empty():
            channel_names.append(child_element.child_value('label'))
            child_element = child_element.next_sibling()

        if not channel_names:
            # Fallback to default names if labels are not provided
            channel_names = [f"status_{i+1}" for i in range(self.num_status_channels)]
            print("No channel labels found in status stream metadata. Using default channel names.")
        else:
            print(f"Retrieved status channel names: {channel_names}")

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

        if self.save_exg or self.save_merged:
            exg_data, ema_timestamps = self.exg_inlet.pull_chunk(timeout=0.0)
            # exg_data shape: (N, num_exg_channels)
            if len(exg_data) > 0:
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

        if self.save_angle or self.save_merged:
            angle_data, angle_timestamps = self.angle_inlet.pull_chunk(timeout=0.0)
            if len(angle_data) > 0:
                angle_data = np.array(angle_data, dtype=self.mp_dtype)
                # Append angle data and timestamps
                self.angle_buffer.extend(angle_data)
                self.angle_timestamp_buffer.extend(np.array(angle_timestamps, dtype=self.timestamp_dtype))
            
        if self.save_status or self.save_merged:
            status_data, status_timestamps = self.status_inlet.pull_chunk(timeout=0.0)
            if len(status_data) > 0:
                status_data = np.array(status_data, dtype=self.mp_dtype)
                # Append angle data and timestamps
                self.status_buffer.extend(status_data)
                self.status_timestamp_buffer.extend(np.array(status_timestamps, dtype=self.timestamp_dtype))

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
                time_to_sleep = self.update_interval
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
        start_time = -1
        end_time = float('inf')
        
        angle_df = None
        if self.angle_buffer.size > 0:
            # Get Angle data
            angle_timestamps = self.angle_timestamp_buffer.get_all()  # shape: (M,)
            angle_samples = self.angle_buffer.get_all()  # shape: (M, num_angle_channels)
            # Match the length of timestamps and samples
            min_len = min(angle_timestamps.shape[0], angle_samples.shape[0])
            angle_timestamps = angle_timestamps[:min_len]
            angle_samples = angle_samples[:min_len]
            # Create DataFrame
            angle_df = pd.DataFrame(angle_samples, index=angle_timestamps, columns=self.mp_channel_names)
            # Update start and end times
            start_time = max(start_time, angle_timestamps[0])
            end_time = min(end_time, angle_timestamps[-1])
            # Set to None if empty
            if angle_df.empty:
                angle_df = None
        
        ema_df = None
        if self.ema_buffer.size > 0:
            # Get EMA data
            ema_timestamps = self.ema_timestamp_buffer.get_all()[self.ema_buffer_start:]  # shape: (N,)
            ema_samples = self.ema_buffer.get_all()[self.ema_buffer_start:]  # shape: (N, len(ema_spans), num_exg_channels)
            self.ema_buffer_start = 0
            # Match the length of timestamps and samples
            min_len = min(ema_timestamps.shape[0], ema_samples.shape[0])
            ema_timestamps = ema_timestamps[:min_len]
            ema_samples = ema_samples[:min_len]
            # Reshape EMA samples to (N, num_exg_channels * len(ema_spans))
            ema_shape = ema_samples.shape
            ema_samples = ema_samples.reshape(-1, ema_shape[-1] * ema_shape[-2])
            # Create DataFrame
            ema_df = pd.DataFrame(ema_samples, index=ema_timestamps, columns=self.exg_span_names)
            # Update start and end times
            start_time = max(start_time, ema_timestamps[0])
            end_time = min(end_time, ema_timestamps[-1])
            # Set to None if empty
            if ema_df.empty:
                ema_df = None
        
        status_df = None
        if self.status_buffer.size > 0:
            # Get Status data
            status_timestamps = self.status_timestamp_buffer.get_all()
            status_samples = self.status_buffer.get_all()
            # Match the length of timestamps and samples
            min_len = min(status_timestamps.shape[0], status_samples.shape[0])
            status_timestamps = status_timestamps[:min_len]
            status_samples = status_samples[:min_len]
            # Create DataFrame
            status_df = pd.DataFrame(status_samples, index=status_timestamps, columns=self.status_channel_names)
            # Update start and end times
            start_time = max(start_time, status_timestamps[0])
            end_time = min(end_time, status_timestamps[-1])
            # Set to None if empty
            if status_df.empty:
                status_df = None
        
        start_time = pd.to_timedelta(start_time, unit='s')
        end_time = pd.to_timedelta(end_time, unit='s')
        ema_df_masked = None    
        if ema_df is not None:
            ema_df.index = pd.to_timedelta(ema_df.index, unit='s')
            ema_mask = (ema_df.index >= start_time) & (ema_df.index <= end_time)
            ema_df_masked = ema_df[ema_mask]
            ema_df.index = (ema_df.index.total_seconds() * 1000).round().astype(int)
            ema_df.index.name = 'timestamp'
            
        angle_df_masked = None
        if angle_df is not None:
            angle_df.index = pd.to_timedelta(angle_df.index, unit='s')
            angle_mask = (angle_df.index >= start_time) & (angle_df.index <= end_time)
            angle_df_masked = angle_df[angle_mask]
            angle_df.index = (angle_df.index.total_seconds() * 1000).round().astype(int)
            angle_df.index.name = 'timestamp'
            
        status_df_masked = None
        if status_df is not None:
            status_df.index = pd.to_timedelta(status_df.index, unit='s')
            status_mask = (status_df.index >= start_time) & (status_df.index <= end_time)
            status_df_masked = status_df[status_mask]
            status_df.index = (status_df.index.total_seconds() * 1000).round().astype(int)
            status_df.index.name = 'timestamp'
            
        merged_df = None
        if self.save_merged and (ema_df_masked is not None and angle_df_masked is not None and status_df_masked is not None):
            # Merge as-of nearest timestamp
            tolerance = min(1/self.angle_rate, 1/self.exg_rate)
            tolerance = pd.to_timedelta(tolerance, unit='s')
            
            merged_df = pd.merge_asof(angle_df_masked, status_df_masked, left_index=True, right_index=True, direction='backward')
            merged_df = pd.merge_asof(merged_df, ema_df_masked, left_index=True, right_index=True, direction='nearest', tolerance=tolerance)
            merged_df.index = (merged_df.index.total_seconds() * 1000).round().astype(int)
            merged_df.index.name = 'timestamp'
        
        return ema_df, angle_df, status_df, merged_df

    def save_data(self):
        """Synchronizes all buffered data and writes to the CSV file."""
        ema_df, angle_df, status_df, merged_df = self.get_data()
        
        if ema_df is None or angle_df is None or merged_df is None:
            print("No data to save.")
            return
        
        # Create CSV directory if it does not exist
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)

        # Save to CSV
        timestamp = int(time.time())
        
        # Save dataframes to CSV files
        if self.save_exg and ema_df is not None:
            if not os.path.exists(f"{self.csv_dir}/exg"):
                os.makedirs(f"{self.csv_dir}/exg")
            ema_filename = f"{self.csv_dir}/exg/data_{timestamp}.csv"
            ema_df.to_csv(ema_filename)
            print(f"EMA Dataframe saved to '{ema_filename}'.")
        
        if self.save_angle and angle_df is not None:
            if not os.path.exists(f"{self.csv_dir}/angle"):
                os.makedirs(f"{self.csv_dir}/angle")
            angle_filename = f"{self.csv_dir}/angle/data_{timestamp}.csv"
            angle_df.to_csv(angle_filename)
            print(f"Angle Dataframe saved to '{angle_filename}'.")
            
        if self.save_status and status_df is not None:
            if not os.path.exists(f"{self.csv_dir}/status"):
                os.makedirs(f"{self.csv_dir}/status")
            status_filename = f"{self.csv_dir}/status/data_{timestamp}.csv"
            status_df.to_csv(status_filename)
            print(f"Status Dataframe saved to '{status_filename}'.")
        
        if self.save_merged and merged_df is not None:
            if not os.path.exists(f"{self.csv_dir}/merged"):
                os.makedirs(f"{self.csv_dir}/merged")
            merged_filename = f"{self.csv_dir}/merged/data_{timestamp}.csv"
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