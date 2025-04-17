from pylsl import resolve_stream, StreamInlet

print("Looking for streams...")
streams = resolve_stream()
if streams:
    print(f"Found {len(streams)} streams.")
    inlet = StreamInlet(streams[0])
    print("Receiving data...")
    while True:
        sample, timestamp = inlet.pull_sample()
        print(sample)
else:
    print("No streams found.")
