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
        if isinstance(idx, int):
            if idx < 0:
                idx += self._size
            if idx < 0 or idx >= self._size:
                raise IndexError("Index out of bounds.")
            return self._buffer[idx]
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self._size)
            if start < 0:
                start += self._size
            if stop < 0:
                stop += self._size
            if start < 0 or start >= self._size or stop < 0 or stop > self._size:
                raise IndexError("Index out of bounds.")
            return self._buffer[start:stop:step]
        elif isinstance(idx, list):
            idx = [i if i >= 0 else i + self._size for i in idx]
            if not all(self.valid_index(i) for i in idx):
                raise IndexError("Index out of bounds.")
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
        return self._buffer[:self._size].shape

    def valid_index(self, idx):
        return -self._size <= idx < self._size

    def need_to_expand(self, new_size):
        return new_size > self.capacity

    def expand(self, new_capacity=None):
        if new_capacity is None:
            new_capacity = self.capacity * 2
        new_buffer = np.zeros((new_capacity, *self._buffer.shape[1:]), dtype=self._dtype)
        new_buffer[:self._size] = self._buffer[:self._size]
        self._buffer = new_buffer

    def append(self, value):
        if self.need_to_expand(self._size + 1):
            new_capacity = max(self.capacity * 2, self._size + 1)
            self.expand(new_capacity)
        self._buffer[self._size] = value
        self._size += 1

    def extend(self, array: np.ndarray):
        if array.shape[1:] != self._buffer.shape[1:]:
            raise ValueError("Array shape does not match buffer shape.")
        array_size = array.shape[0]
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

class RawRecorder:
    """
    Class to record raw EXG and Angle data from LSL streams and save to CSV files.
    """
    def __init__(self,
                 exg_stream_name: str = "raw_exg",
                 finger_stream_name: str = "finger_percentages",
                 status_stream_name: str = "finger_status",
                 csv_dir: str = "data/session",
                 save_exg: bool = True,
                 save_angle: bool = True,
                 save_status: bool = True,
                 init_capacity_s: float = 1024,
                 update_interval: float = 0.01):
        """
        Initializes the recorder.

        :param exg_stream_name: Name of the EXG stream.
        :param finger_stream_name: Name of the Angle stream.
        :param csv_dir: Output CSV files directory.
        """
        self.csv_dir = csv_dir
        self.save_exg = save_exg
        self.save_angle = save_angle
        self.save_status = save_status
        self.update_interval = update_interval
        self.subtract_time = float('inf')
        
        self.proc_flags = pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_monotonize
        timestamp_shape = ()

        # EXG Stream
        if self.save_exg:
            self.exg_stream_name = exg_stream_name
            print("Resolving EXG stream...")
            exg_streams = resolve_byprop('name', self.exg_stream_name, timeout=5)
            if not exg_streams:
                raise RuntimeError(f"No EXG stream found with name '{self.exg_stream_name}'.")
            self.exg_inlet = StreamInlet(exg_streams[0], max_buflen=1024, processing_flags=self.proc_flags)
            exg_info = self.exg_inlet.info()
            self.exg_rate = exg_info.nominal_srate()
            if self.exg_rate <= 0:
                raise ValueError("EXG stream sampling rate is not set or zero.")
            self.num_exg_channels = exg_info.channel_count()
            if self.num_exg_channels <= 0:
                raise ValueError("EXG stream channel_count is not set or zero.")
            
            init_capacity_exg = int(init_capacity_s * self.exg_rate)
            exg_shape = (self.num_exg_channels,)
            self.exg_buffer = NumpyBuffer(init_capacity=init_capacity_exg, shape=exg_shape, dtype=self.exg_dtype)
            self.exg_timestamp_buffer = NumpyBuffer(init_capacity=init_capacity_exg, shape=(), dtype=self.timestamp_dtype)
            
            self.exg_channel_names = [f"ch_{i+1}" for i in range(self.num_exg_channels)]
            print(f'EXG channel names: {self.exg_channel_names}')

        # Angle Stream
        if self.save_angle:
            self.finger_stream_name = finger_stream_name
            print("Resolving angle stream...")
            angle_streams = resolve_byprop('name', self.finger_stream_name, timeout=5)
            if not angle_streams:
                raise RuntimeError(f"No angle stream found with name '{self.finger_stream_name}'.")
            self.angle_inlet = StreamInlet(angle_streams[0], max_buflen=1024, processing_flags=self.proc_flags)
            angle_info = self.angle_inlet.info()
            self.angle_rate = angle_info.nominal_srate()
            if self.angle_rate <= 0:
                raise ValueError("Angle stream sampling rate is not set or zero.")
            self.num_angle_channels = angle_info.channel_count()
            if self.num_angle_channels <= 0:
                raise ValueError("Angle stream channel_count is not set or zero.")
            
            self.angle_channel_names = self._get_angle_channel_names(angle_info)
            angle_dtype = np.float32
            init_capacity_angle = int(init_capacity_s * self.angle_rate)
            angle_shape = (self.num_angle_channels,)
            self.angle_buffer = NumpyBuffer(init_capacity=init_capacity_angle, shape=angle_shape, dtype=angle_dtype)
            self.angle_timestamp_buffer = NumpyBuffer(init_capacity=init_capacity_angle, shape=(), dtype=self.timestamp_dtype)
            
            print(f'Angle channel names: {self.angle_channel_names}')
        
        # Status Stream
        if self.save_status:
            self.status_stream_name = status_stream_name
            print("Resolving status stream...")
            status_streams = resolve_byprop('name', self.status_stream_name, timeout=5)
            if not status_streams:
                raise RuntimeError(f"No status stream found with name '{self.status_stream_name}'.")
            self.status_inlet = StreamInlet(status_streams[0], max_buflen=1024, processing_flags=self.proc_flags)
            status_info = self.status_inlet.info()
            self.status_rate = status_info.nominal_srate()
            if self.status_rate <= 0:
                raise ValueError("Status stream sampling rate is not set or zero.")
            self.num_status_channels = status_info.channel_count()
            if self.num_status_channels <= 0:
                raise ValueError("Status stream channel_count is not set or zero.")
            
            self.status_channel_names = self._get_status_channel_names(status_info)
            status_dtype = np.float32
            init_capacity_status = int(init_capacity_s * self.status_rate)
            status_shape = (self.num_status_channels,)
            self.status_buffer = NumpyBuffer(init_capacity=init_capacity_status, shape=status_shape, dtype=status_dtype)
            self.status_timestamp_buffer = NumpyBuffer(init_capacity=init_capacity_status, shape=timestamp_shape, dtype=self.timestamp_dtype)
            
            
    def _connect_lsl_stream(self, stream_name, init_capacity_s):
        """
        Connect to an LSL stream and return the inlet and buffer objects.
        """
        
        class StreamInfo:
            """
            _summary_ : Class to store stream information.
            """
            def __init__(self, inlet, buffer, timestamp_buffer, rate, num_channels):
                self.inlet = inlet
                self.buffer = buffer
                self.timestamp_buffer = timestamp_buffer
                self.rate = rate
                self.num_channels = num_channels
        
        print(f"Resolving {stream_name} stream...")
        streams = resolve_byprop('name', stream_name, timeout=5)
        if not streams:
            raise RuntimeError(f"No {stream_name} stream found with name '{stream_name}'.")
        inlet = StreamInlet(streams[0], max_buflen=1024, processing_flags=self.proc_flags)
        info = inlet.info()
        rate = info.nominal_srate()
        if rate <= 0:
            raise ValueError(f"{stream_name} stream sampling rate is not set or zero.")
        num_channels = info.channel_count()
        if num_channels <= 0:
            raise ValueError(f"{stream_name} stream channel_count is not set or zero.")
        shape = (num_channels,)
        
        buffer = NumpyBuffer(init_capacity=int(init_capacity_s * rate), shape=shape, dtype=np.float32)
        timestamp_buffer = NumpyBuffer(init_capacity=int(init_capacity_s * rate), shape=(), dtype=np.float64)
        
        return StreamInfo(inlet, buffer, timestamp_buffer, rate, num_channels)
    
    def _get_angle_channel_names(self, angle_info):
        angle_desc = angle_info.desc()
        child_element = angle_desc.child('channels').child('channel')
        channel_names = []
        while not child_element.empty():
            channel_names.append(child_element.child_value('label'))
            child_element = child_element.next_sibling()
        if not channel_names:
            channel_names = [f"angle_{i+1}" for i in range(self.num_angle_channels)]
            print("No channel labels found in angle stream metadata. Using default channel names.")
        else:
            print(f"Retrieved angle channel names: {channel_names}")
        return channel_names
    
    def _get_status_channel_names(self, status_info):
        status_desc = status_info.desc()
        child_element = status_desc.child('channels').child('channel')
        channel_names = []
        while not child_element.empty():
            channel_names.append(child_element.child_value('label'))
            child_element = child_element.next_sibling()
        if not channel_names:
            channel_names = [f"status_{i+1}" for i in range(self.num_status_channels)]
            print("No channel labels found in status stream metadata. Using default channel names.")
        else:
            print(f"Retrieved status channel names: {channel_names}")
        return channel_names
    
    @property
    def exg_dtype(self):
        return np.float32

    @property
    def timestamp_dtype(self):
        return np.float64

    @property
    def duration_exg_s(self):
        return self.exg_buffer.size / self.exg_rate

    @property
    def duration_angle_s(self):
        return self.angle_buffer.size / self.angle_rate

    def update(self):
        # Pull data from EXG, Angle, and Status streams
        if self.save_exg:
            exg_data, exg_timestamps = self.exg_inlet.pull_chunk(timeout=0.0)
            if len(exg_data) > 0:
                raw_exg_data = np.array(exg_data, dtype=self.exg_buffer.dtype)
                self.exg_buffer.extend(raw_exg_data)
                self.exg_timestamp_buffer.extend(np.array(exg_timestamps, dtype=self.timestamp_dtype))
        
        if self.save_angle:
            angle_data, angle_timestamps = self.angle_inlet.pull_chunk(timeout=0.0)
            if len(angle_data) > 0:
                angle_data = np.array(angle_data, dtype=self.angle_buffer.dtype)
                self.angle_buffer.extend(angle_data)
                self.angle_timestamp_buffer.extend(np.array(angle_timestamps, dtype=self.timestamp_dtype))
            
        if self.save_status:
            status_data, status_timestamps = self.status_inlet.pull_chunk(timeout=0.0)
            if len(status_data) > 0:
                status_data = np.array(status_data, dtype=self.status_buffer.dtype)
                self.status_buffer.extend(status_data)
                self.status_timestamp_buffer.extend(np.array(status_timestamps, dtype=self.timestamp_dtype))

    def collect_and_save_loop(self):
        print("Starting data collection loop. Press Ctrl+C to stop and save data.")
        try:
            while True:
                self.update()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Preparing to save data...")
            self.save_data()
            
    def get_data(self):
        start_time = -1
        end_time = pylsl.local_clock()
        
        exg_df = None
        if self.save_exg and self.exg_buffer.size > 0:
            exg_timestamps = self.exg_timestamp_buffer.get_all()
            exg_samples = self.exg_buffer.get_all()
            min_len = min(exg_timestamps.shape[0], exg_samples.shape[0])
            exg_timestamps = exg_timestamps[:min_len]
            exg_samples = exg_samples[:min_len]
            exg_df = pd.DataFrame(exg_samples, index=exg_timestamps, columns=self.exg_channel_names)
            self.subtract_time = min(self.subtract_time, exg_timestamps[0])
            start_time = max(start_time, exg_timestamps[0])
            end_time = min(end_time, exg_timestamps[-1])
            if exg_df.empty:
                exg_df = None
                
        angle_df = None
        if self.save_angle and self.angle_buffer.size > 0:
            angle_timestamps = self.angle_timestamp_buffer.get_all()
            angle_samples = self.angle_buffer.get_all()
            min_len = min(angle_timestamps.shape[0], angle_samples.shape[0])
            angle_timestamps = angle_timestamps[:min_len]
            angle_samples = angle_samples[:min_len]
            angle_df = pd.DataFrame(angle_samples, index=angle_timestamps, columns=self.angle_channel_names)
            self.subtract_time = min(self.subtract_time, angle_timestamps[0])
            start_time = max(start_time, angle_timestamps[0])
            end_time = min(end_time, angle_timestamps[-1])
            if angle_df.empty:
                angle_df = None
                
        status_df = None
        if self.save_status and self.status_buffer.size > 0:
            status_timestamps = self.status_timestamp_buffer.get_all()
            status_samples = self.status_buffer.get_all()
            min_len = min(status_timestamps.shape[0], status_samples.shape[0])
            status_timestamps = status_timestamps[:min_len]
            status_samples = status_samples[:min_len]
            status_df = pd.DataFrame(status_samples, index=status_timestamps, columns=self.status_channel_names)
            self.subtract_time = min(self.subtract_time, status_timestamps[0])
            if status_df.empty:
                status_df = None
                
        if self.subtract_time == float('inf'):
            self.subtract_time = 0
        
        subtract_time = pd.to_timedelta(self.subtract_time, unit='s')
        start_time = pd.to_timedelta(start_time, unit='s')
        end_time = pd.to_timedelta(end_time, unit='s')
        
        if exg_df is not None:
            exg_df.index = pd.to_timedelta(exg_df.index, unit='s')
            exg_df.index -= subtract_time
            exg_df.index = (exg_df.index.total_seconds() * 1000).round().astype(int)
            exg_df.index.name = 'timestamp'
            
        if angle_df is not None:
            angle_df.index = pd.to_timedelta(angle_df.index, unit='s')
            angle_df.index -= subtract_time
            angle_df.index = (angle_df.index.total_seconds() * 1000).round().astype(int)
            angle_df.index.name = 'timestamp'
            
        if status_df is not None:
            status_df.index = pd.to_timedelta(status_df.index, unit='s')
            status_df.index -= subtract_time
            status_df.index = (status_df.index.total_seconds() * 1000).round().astype(int)
            status_df.index.name = 'timestamp'
        
        return exg_df, angle_df, status_df

    def save_data(self):
        exg_df, angle_df, status_df = self.get_data()
        
        if exg_df is None and angle_df is None:
            print("No data to save.")
            return
        
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        
        timestamp = int(time.time())
        
        if self.save_exg and exg_df is not None:
            exg_dir = os.path.join(self.csv_dir, "exg")
            if not os.path.exists(exg_dir):
                os.makedirs(exg_dir)
            exg_filename = os.path.join(exg_dir, f"data_{timestamp}.csv")
            exg_df.to_csv(exg_filename)
            print(f"EXG Dataframe saved to '{exg_filename}'.")
        
        if self.save_angle and angle_df is not None:
            angle_dir = os.path.join(self.csv_dir, "angle")
            if not os.path.exists(angle_dir):
                os.makedirs(angle_dir)
            angle_filename = os.path.join(angle_dir, f"data_{timestamp}.csv")
            angle_df.to_csv(angle_filename)
            print(f"Angle Dataframe saved to '{angle_filename}'.")
            
        if self.save_status and status_df is not None:
            status_dir = os.path.join(self.csv_dir, "status")
            if not os.path.exists(status_dir):
                os.makedirs(status_dir)
            status_filename = os.path.join(status_dir, f"data_{timestamp}.csv")
            status_df.to_csv(status_filename)
            print(f"Status Dataframe saved to '{status_filename}'.")
        
        print("Data saved successfully.")


def main():
    recorder = RawRecorder(
        exg_stream_name="raw_exg",
        finger_stream_name="finger_percentages",
        csv_dir="data/s_0"
    )
    recorder.collect_and_save_loop()


if __name__ == "__main__":
    main()
