import numpy as np
from pylsl import StreamInlet, resolve_byprop, proc_ALL
import time
import pandas as pd

class NumpyBuffer:
    def __init__(self, capacity, shape, dtype):
        self.capacity = capacity
        self.shape = shape
        self.dtype = dtype
        self.buffer = np.zeros((capacity, *shape), dtype=dtype)
        self.timestamps = np.zeros(capacity, dtype=np.float64)
        self.size = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0:
                idx += self.size
            if idx < 0 or idx >= self.size:
                raise IndexError("Index out of bounds.")
            return self.timestamps[idx], self.buffer[idx]
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.size)
            return (self.timestamps[start:stop:step],
                    self.buffer[start:stop:step])
        else:
            raise TypeError("Invalid index type.")

    def expand(self):
        new_capacity = self.capacity * 2
        new_buffer = np.zeros((new_capacity, *self.shape), dtype=self.dtype)
        new_timestamps = np.zeros(new_capacity, dtype=np.float64)
        new_buffer[:self.size] = self.buffer[:self.size]
        new_timestamps[:self.size] = self.timestamps[:self.size]
        self.buffer = new_buffer
        self.timestamps = new_timestamps
        self.capacity = new_capacity

    def extend(self, timestamps, samples):
        new_size = len(samples)
        if self.size + new_size > self.capacity:
            while self.size + new_size > self.capacity:
                self.expand()
        self.buffer[self.size:self.size+new_size] = samples
        self.timestamps[self.size:self.size+new_size] = timestamps
        self.size += new_size

    def get_all(self):
        data = (self.timestamps[:self.size].copy(),
                self.buffer[:self.size].copy())
        self.size = 0  # Clear buffer after fetching
        return data

class EMGAngleRecorder:
    def __init__(self, 
                 emg_stream_name="filtered_exg", 
                 angle_stream_name="FingerPercentages",
                 buffer_seconds=60*10,  # Increased buffer to accommodate all data
                 csv_filename="data/emg_mp",
                 finger_thresholds = ((0.825, 0.9), (0.7, 0.875), 
                                     (0.75, 0.875), (0.7, 0.8), 
                                     (0.7, 0.825))):
        """
        Initializes the recorder.

        :param emg_stream_name: Name of the EMG stream.
        :param angle_stream_name: Name of the Angle stream.
        :param buffer_seconds: Duration of data to buffer.
        :param csv_filename: Output CSV file name.
        :param finger_thresholds: Thresholds for angle data.
        """
        self.emg_stream_name = emg_stream_name
        self.angle_stream_name = angle_stream_name
        self.buffer_seconds = buffer_seconds
        self.csv_filename = csv_filename
        self.finger_thresholds = np.array(finger_thresholds)

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
        self.angle_rate = angle_info.nominal_srate()
        if self.angle_rate <= 0:
            raise ValueError("Angle stream sampling rate is not set or zero.")
        self.num_angle_channels = angle_info.channel_count()
        if self.num_angle_channels <= 0:
            raise ValueError("Angle stream channel_count is not set or zero.")
        
        # Extract EMG channel names
        self.emg_channel_names = [f"ch_{i+1}" for i in range(self.num_emg_channels)]
            
        # Extract Angle channel names
        self.mp_channel_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            
        print(f'EMG channel names: {self.emg_channel_names}')
        print(f'Angle channel names: {self.mp_channel_names}')

        # Calculate buffer capacity
        self.emg_capacity = int(buffer_seconds * self.emg_rate)
        self.angle_capacity = int(buffer_seconds * self.angle_rate)

        # Initialize buffers
        self.emg_buffer = NumpyBuffer(self.emg_capacity, (self.num_emg_channels,), dtype=np.float32)
        self.angle_buffer = NumpyBuffer(self.angle_capacity, (self.num_angle_channels,), dtype=np.float32)

    def collect_data(self):
        """Continuously collects data from the streams until interrupted."""
        print("Starting data collection. Press Ctrl+C to stop and save data.")
        try:
            while True:
                # Pull data from EMG
                emg_data, emg_timestamps = self.emg_inlet.pull_chunk(timeout=0.0)
                if emg_data:
                    emg_data = np.array(emg_data, dtype=np.float32)
                    emg_timestamps = np.array(emg_timestamps, dtype=np.float64)
                    self.emg_buffer.extend(emg_timestamps, emg_data)

                # Pull data from Angle
                angle_data, angle_timestamps = self.angle_inlet.pull_chunk(timeout=0.0)
                if angle_data:
                    angle_data = np.array(angle_data, dtype=np.float32)
                    angle_timestamps = np.array(angle_timestamps, dtype=np.float64)
                    # Clip and normalize angle data
                    angle_data = np.clip(angle_data, self.finger_thresholds[:, 0], self.finger_thresholds[:, 1])
                    angle_data = (angle_data - self.finger_thresholds[:, 0]) / (self.finger_thresholds[:, 1] - self.finger_thresholds[:, 0])
                    self.angle_buffer.extend(angle_timestamps, angle_data)

                # Sleep briefly to prevent CPU overuse
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Preparing to save data...")
            self.save_data()

    def save_data(self):
        """Synchronizes all buffered data and writes to the CSV file."""
        # Fetch all data from buffers
        emg_timestamps, emg_samples = self.emg_buffer.get_all()
        angle_timestamps, angle_samples = self.angle_buffer.get_all()

        if emg_samples.size == 0 and angle_samples.size == 0:
            print("No data collected. Exiting without saving.")
            return
        
        start = max(emg_timestamps[0], angle_timestamps[0])
        end = min(emg_timestamps[-1], angle_timestamps[-1])
        
        emg_mask = (emg_timestamps >= start) & (emg_timestamps <= end)
        angle_mask = (angle_timestamps >= start) & (angle_timestamps <= end)
        
        emg_timestamps = emg_timestamps[emg_mask]
        emg_samples = emg_samples[emg_mask]
        angle_timestamps = angle_timestamps[angle_mask]
        angle_samples = angle_samples[angle_mask]
        # Synchronize data based on nearest timestamps using Pandas
        
        emg_df = pd.DataFrame(emg_samples, index=emg_timestamps, columns=self.emg_channel_names)
        angle_df = pd.DataFrame(angle_samples, index=angle_timestamps, columns=self.mp_channel_names)
        # Set timestamps as timedelta type
        emg_df.index = pd.to_timedelta(emg_df.index, unit='s')
        angle_df.index = pd.to_timedelta(angle_df.index, unit='s')
        
        # Merge as of the nearest timestamp based
        merged_df = pd.merge_asof(angle_df, emg_df, left_index=True, right_index=True, direction='nearest')
        
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
    recorder = EMGAngleRecorder(
        emg_stream_name="filtered_exg",
        angle_stream_name="FingerPercentages",
        csv_filename="data/s_0/emg_mp",
        buffer_seconds=60*30,  # Adjust buffer based on how long you want to record
        finger_thresholds=((0.85, 0.9), (0.7, 0.875), 
                          (0.75, 0.875), (0.725, 0.8), 
                          (0.725, 0.825))
    )
    recorder.collect_data()

if __name__ == "__main__":
    main()