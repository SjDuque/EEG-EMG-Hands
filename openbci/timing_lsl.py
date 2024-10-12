import time
import threading
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
import pylsl

def emg_data_producer(stream_name, stream_type, channel_count, nominal_srate, stop_event):
    """Simulates an EMG data stream with embedded time.time() in the samples."""
    # Update channel count to include time.time()
    total_channels = channel_count + 1  # Extra channel for time.time()
    info = StreamInfo(stream_name, stream_type, total_channels, nominal_srate, 'double64', 'emg_test_stream')
    outlet = StreamOutlet(info, chunk_size=1, max_buffered=360)

    print(f"EMG data producer started: {stream_name}")

    # Use high-resolution timer
    next_sample_time = time.perf_counter()
    sample_interval = 1.0 / nominal_srate

    while not stop_event.is_set():
        # Generate dummy data
        sample = [float(i) for i in range(channel_count)]
        # Append time.time() to sample
        current_time = time.time()
        sample.append(current_time)

        # Send sample without specifying timestamp (uses LSL local clock)
        outlet.push_sample(sample)

        # Calculate the time until the next sample should be sent
        next_sample_time += sample_interval
        sleep_duration = next_sample_time - time.perf_counter()

        if sleep_duration > 0:
            time.sleep(sleep_duration)
        else:
            # We're behind schedule; adjust next sample time
            next_sample_time = time.perf_counter()

    print("EMG data producer stopped.")

def collect_emg_data(inlet, emg_data_list, emg_channel_count, stop_event):
    """Collects EMG data and compares converted LSL timestamps with embedded time.time()."""
    sample_count = 0

    # Calculate the offset at the start
    offset = time.time() - pylsl.local_clock()
    print(f"Offset (Unix time - LSL time): {offset} seconds")

    while not stop_event.is_set():
        try:
            sample, lsl_timestamp = inlet.pull_sample(timeout=1.0)
            if sample is None:
                continue
            sample_count += 1

            # Convert LSL timestamp to Unix timestamp
            unix_timestamp = lsl_timestamp + offset

            # Extract the embedded time.time() value from the sample
            embedded_time = sample[-1]  # The last element is the embedded time.time()
            emg_data = sample[:-1]      # The rest is the EMG data

            # Calculate the difference between converted LSL timestamp and embedded time
            time_difference = unix_timestamp - embedded_time

            # Optional: Debugging statements
            if sample_count % 100 == 0:
                print(f"Sample {sample_count}")
                print(f"LSL Timestamp: {lsl_timestamp}")
                print(f"Unix Timestamp: {unix_timestamp}")
                print(f"Embedded time.time(): {embedded_time}")
                print(f"Time Difference (Unix Timestamp - Embedded time.time()): {time_difference} seconds")
                print("-" * 50)

            data_row = {'timestamp': unix_timestamp}
            for i in range(emg_channel_count - 1):
                data_row[f'emg_channel_{i + 1}'] = emg_data[i]
            data_row['embedded_time'] = embedded_time
            emg_data_list.append(data_row)
        except Exception as e:
            print(f"Error in EMG data collection: {e}")
            continue

    print("EMG data collector stopped.")

def main():
    # Stream parameters
    stream_name = 'TestEMGStream'
    stream_type = 'EMG'
    channel_count = 8  # Number of EMG channels
    nominal_srate = 1000  # Sampling rate in Hz

    # Create a stop event for threading
    stop_event = threading.Event()

    # Start the EMG data producer in a separate thread
    producer_thread = threading.Thread(
        target=emg_data_producer,
        args=(stream_name, stream_type, channel_count, nominal_srate, stop_event)
    )
    producer_thread.start()

    # Allow some time for the stream to initialize
    time.sleep(1)

    # Resolve the EMG stream
    print("Resolving EMG stream...")
    streams = resolve_stream('name', stream_name)
    if not streams:
        print("No EMG stream found.")
        stop_event.set()
        producer_thread.join()
        return

    # Create an inlet without proc_clocksync
    inlet = StreamInlet(streams[0], processing_flags=0)

    # Prepare to collect data
    emg_data_list = []
    # Adjust emg_channel_count to include the embedded time channel
    total_channels = channel_count + 1
    collector_thread = threading.Thread(
        target=collect_emg_data,
        args=(inlet, emg_data_list, total_channels, stop_event)
    )
    collector_thread.start()

    # Collect data for a specified duration
    collection_duration = 5  # seconds
    print(f"Collecting data for {collection_duration} seconds...")
    time.sleep(collection_duration)

    # Stop the data collection
    stop_event.set()
    collector_thread.join()
    producer_thread.join()

    # Verify the collected data
    print(f"Collected {len(emg_data_list)} samples.")
    if emg_data_list:
        # Check the time differences
        first_sample = emg_data_list[0]
        last_sample = emg_data_list[-1]
        time_span = last_sample['timestamp'] - first_sample['timestamp']
        print(f"Time span of collected data: {time_span} seconds")

        # Calculate expected number of samples
        expected_samples = int(nominal_srate * collection_duration)
        print(f"Expected number of samples: {expected_samples}")

        # Verify sampling rate
        actual_srate = len(emg_data_list) / time_span
        print(f"Actual sampling rate: {actual_srate} Hz")

        # Verify timestamps are synchronized
        last_timestamp = last_sample['timestamp']
        embedded_time = last_sample['embedded_time']
        time_difference = last_timestamp - embedded_time
        print(f"Time difference between last timestamp and embedded time: {time_difference} seconds")

        if abs(time_difference) < 0.01:
            print("Timestamps are correctly synchronized with embedded time.")
        else:
            print("Timestamps are not correctly synchronized.")

if __name__ == "__main__":
    main()
