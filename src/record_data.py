import numpy as np
from pylsl import StreamInlet, resolve_stream, resolve_byprop
import threading
import time
import csv
import sys

class NumpyBuffer:
    def __init__(self, capacity, shape, dtype):
        self.capacity = capacity
        self.shape = shape
        self.dtype = dtype
        self.buffer = np.zeros((capacity, *shape), dtype=dtype)
        self.timestamps = np.zeros(capacity, dtype=np.float64)
        self.size = 0
        self.lock = threading.Lock()
        
    def __len__(self):
        with self.lock:
            return self.size
        
    def __getitem__(self, idx):
        with self.lock:
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
        with self.lock:
            new_capacity = self.capacity * 2
            new_buffer = np.zeros((new_capacity, *self.shape), dtype=self.dtype)
            new_timestamps = np.zeros(new_capacity, dtype=np.float64)
            new_buffer[:self.size] = self.buffer[:self.size]
            new_timestamps[:self.size] = self.timestamps[:self.size]
            self.buffer = new_buffer
            self.timestamps = new_timestamps
            self.capacity = new_capacity
        
    def extend(self, timestamps, samples):
        with self.lock:
            new_size = len(samples)
            if self.size + new_size > self.capacity:
                while self.size + new_size > self.capacity:
                    self.expand()
            self.buffer[self.size:self.size+new_size] = samples
            self.timestamps[self.size:self.size+new_size] = timestamps
            self.size += new_size
        
    def get_all(self):
        with self.lock:
            data = (self.timestamps[:self.size].copy(),
                    self.buffer[:self.size].copy())
            self.size = 0  # Clear buffer after fetching
            return data

class EMGAngleRecorder:
    def __init__(self, 
                 emg_stream_name="filtered_exg", 
                 angle_stream_name="FingerPercentages",
                 buffer_seconds=5, 
                 csv_filename="emg_angle_data.csv",
                 finger_thresholds = ((0.825, 0.9), (0.7, 0.875), 
                                     (0.75, 0.875), (0.7, 0.8), 
                                     (0.7, 0.825)),
                 write_interval=1.0):
        """
        Initializes the recorder.

        :param emg_stream_name: Name of the EMG stream.
        :param angle_stream_name: Name of the Angle stream.
        :param buffer_seconds: Duration of data to buffer.
        :param csv_filename: Output CSV file name.
        :param finger_thresholds: Thresholds for angle data.
        :param write_interval: Time interval (in seconds) between writing to CSV.
        """
        self.emg_stream_name = emg_stream_name
        self.angle_stream_name = angle_stream_name
        self.buffer_seconds = buffer_seconds
        self.csv_filename = csv_filename
        self.finger_thresholds = np.array(finger_thresholds)
        self.write_interval = write_interval  # seconds

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
        self.emg_inlet = StreamInlet(emg_streams[0], max_buflen=2)
        self.angle_inlet = StreamInlet(angle_streams[0], max_buflen=2)

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

        # Calculate buffer capacity
        self.emg_capacity = int(buffer_seconds * self.emg_rate)
        self.angle_capacity = int(buffer_seconds * self.angle_rate)

        # Initialize buffers
        self.emg_buffer = NumpyBuffer(self.emg_capacity, (self.num_emg_channels,), dtype=np.float32)
        self.angle_buffer = NumpyBuffer(self.angle_capacity, (self.num_angle_channels,), dtype=np.float32)

        # Threading
        self.stop_flag = False
        self.data_lock = threading.Lock()
        self.collect_thread = threading.Thread(target=self.collect_data, daemon=True)
        self.write_thread = threading.Thread(target=self.write_data, daemon=True)

        # Initialize CSV
        self.initialize_csv()

    def initialize_csv(self):
        """Initializes the CSV file with headers."""
        with open(self.csv_filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            emg_headers = [f"EMG_Channel_{i+1}" for i in range(self.num_emg_channels)]
            angle_headers = [f"Angle_Channel_{i+1}" for i in range(self.num_angle_channels)]
            headers = ["Timestamp"] + emg_headers + angle_headers
            writer.writerow(headers)
        print(f"CSV file '{self.csv_filename}' initialized.")

    def collect_data(self):
        """Continuously collects data from the streams."""
        while not self.stop_flag:
            # Pull data from EMG
            emg_data, emg_timestamps = self.emg_inlet.pull_chunk(timeout=0.0)
            if emg_data:
                emg_data = np.array(emg_data, dtype=np.float32)
                emg_timestamps = np.array(emg_timestamps, dtype=np.float64)
                with self.data_lock:
                    self.emg_buffer.extend(emg_timestamps, emg_data)

            # Pull data from Angle
            angle_data, angle_timestamps = self.angle_inlet.pull_chunk(timeout=0.0)
            if angle_data:
                angle_data = np.array(angle_data, dtype=np.float32)
                angle_timestamps = np.array(angle_timestamps, dtype=np.float64)
                # Clip and normalize angle data
                angle_data = np.clip(angle_data, self.finger_thresholds[:, 0], self.finger_thresholds[:, 1])
                angle_data = (angle_data - self.finger_thresholds[:, 0]) / (self.finger_thresholds[:, 1] - self.finger_thresholds[:, 0])
                with self.data_lock:
                    self.angle_buffer.extend(angle_timestamps, angle_data)
            
            time.sleep(0.001)  # Prevent CPU overuse

    def write_data(self):
        """Periodically writes buffered data to the CSV file."""
        while not self.stop_flag:
            time.sleep(self.write_interval)
            with self.data_lock:
                emg_timestamps, emg_samples = self.emg_buffer.get_all()
                angle_timestamps, angle_samples = self.angle_buffer.get_all()
            
            if emg_samples.size == 0 and angle_samples.size == 0:
                continue  # Nothing to write

            # Synchronize data based on timestamps
            combined_data = self.synchronize_data(emg_timestamps, emg_samples,
                                                  angle_timestamps, angle_samples)

            if combined_data:
                self.append_to_csv(combined_data)

    def synchronize_data(self, emg_timestamps, emg_samples, angle_timestamps, angle_samples):
        """
        Synchronizes EMG and Angle data based on timestamps.

        Returns a list of rows to write to CSV.
        Each row contains:
        [Timestamp, EMG_Channel_1, ..., EMG_Channel_N, Angle_Channel_1, ..., Angle_Channel_M]
        """
        rows = []
        if emg_samples.size > 0 and angle_samples.size > 0:
            # Find overlapping timestamps
            start_time = max(emg_timestamps[0], angle_timestamps[0])
            end_time = min(emg_timestamps[-1], angle_timestamps[-1])
            if start_time >= end_time:
                return rows  # No overlapping data

            # Find indices for overlapping data
            emg_start_idx = np.searchsorted(emg_timestamps, start_time, side='left')
            emg_end_idx = np.searchsorted(emg_timestamps, end_time, side='right')
            angle_start_idx = np.searchsorted(angle_timestamps, start_time, side='left')
            angle_end_idx = np.searchsorted(angle_timestamps, end_time, side='right')

            emg_subset = emg_samples[emg_start_idx:emg_end_idx]
            emg_subset_ts = emg_timestamps[emg_start_idx:emg_end_idx]
            angle_subset = angle_samples[angle_start_idx:angle_end_idx]
            angle_subset_ts = angle_timestamps[angle_start_idx:angle_end_idx]

            # For simplicity, we'll pair the closest EMG and Angle samples
            min_len = min(len(emg_subset_ts), len(angle_subset_ts))
            for i in range(min_len):
                row_timestamp = max(emg_subset_ts[i], angle_subset_ts[i])
                emg_row = emg_subset[i]
                angle_row = angle_subset[i]
                row = [row_timestamp] + emg_row.tolist() + angle_row.tolist()
                rows.append(row)
        elif emg_samples.size > 0:
            for ts, sample in zip(emg_timestamps, emg_samples):
                row = [ts] + sample.tolist() + ["" for _ in range(self.num_angle_channels)]
                rows.append(row)
        elif angle_samples.size > 0:
            for ts, sample in zip(angle_timestamps, angle_samples):
                row = [ts] + ["" for _ in range(self.num_emg_channels)] + sample.tolist()
                rows.append(row)
        return rows

    def append_to_csv(self, rows):
        """Appends rows of data to the CSV file."""
        with open(self.csv_filename, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for row in rows:
                writer.writerow(row)
        print(f"Appended {len(rows)} rows to '{self.csv_filename}'.")

    def start(self):
        """Starts the data collection and writing threads."""
        print("Starting data collection...")
        self.collect_thread.start()
        self.write_thread.start()
        print("Recording started. Press Ctrl+C to stop.")

    def stop(self):
        """Stops the threads and ensures all data is written."""
        print("Stopping recording...")
        self.stop_flag = True
        self.collect_thread.join()
        self.write_thread.join()
        # Write any remaining data
        with self.data_lock:
            emg_timestamps, emg_samples = self.emg_buffer.get_all()
            angle_timestamps, angle_samples = self.angle_buffer.get_all()
        combined_data = self.synchronize_data(emg_timestamps, emg_samples,
                                             angle_timestamps, angle_samples)
        if combined_data:
            self.append_to_csv(combined_data)
        print("Recording stopped and data saved.")

def main():
    recorder = EMGAngleRecorder(
        emg_stream_name="filtered_exg",
        angle_stream_name="FingerPercentages",
        buffer_seconds=60*10,
        csv_filename="emg_angle_data.csv",
        finger_thresholds=((0.825, 0.9), (0.7, 0.875), 
                          (0.75, 0.875), (0.7, 0.8), 
                          (0.7, 0.825)),
        write_interval=1.0  # Write every second
    )
    try:
        recorder.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting...")
        recorder.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()
