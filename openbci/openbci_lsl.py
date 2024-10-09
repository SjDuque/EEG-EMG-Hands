from pylsl import StreamInlet, resolve_stream
import time

# Resolve all streams with type 'EMG'
print("Looking for all available EMG streams...")
streams = resolve_stream('type', 'EMG')

# Check if any streams were found
if len(streams) == 0:
    print("No EMG streams found.")
else:
    print(f"{len(streams)} EMG stream(s) found:")
    for idx, stream in enumerate(streams):
        print(f"Stream {idx + 1}:")
        print(f"  Name: {stream.name()}")
        print(f"  Type: {stream.type()}")
        print(f"  Channel Count: {stream.channel_count()}")
        print(f"  Source ID: {stream.source_id()}")
        print(f"  Nominal Sampling Rate: {stream.nominal_srate()} Hz")
        print(f"  Hostname: {stream.hostname()}")

    # Select the first available stream
    inlet = StreamInlet(streams[0])
    print("\nUsing the first EMG stream...")

    # Initialize variables for tracking refresh rate and loop count
    loop_count = 0
    start_time = time.time()
    last_print_time = start_time  # Track time since the last print of refresh rate

    while True:
        # Get EMG data sample and its timestamp
        sample, timestamp = inlet.pull_sample()

        # Convert the timestamp to a human-readable format (hh:mm:ss)
        formatted_time = time.strftime('%H:%M:%S', time.localtime(timestamp))

        # Increment loop count
        loop_count += 1

        # Get current time
        current_time = time.time()

        # Calculate elapsed time since last refresh rate print
        elapsed_time = current_time - last_print_time

        # Check if one second has passed since the last refresh rate print
        if elapsed_time >= 1.0:
            # Calculate the refresh rate (loops per second)
            refresh_rate = loop_count / elapsed_time

            # Print the sample size, formatted time, and per-second refresh rate
            print("Sample Size: ", len(sample))
            print("Timestamp: ", formatted_time)
            print("Refresh Rate: {:.2f} Hz".format(refresh_rate))

            # Reset loop count and update the last print time to current time
            loop_count = 0
            last_print_time = current_time
