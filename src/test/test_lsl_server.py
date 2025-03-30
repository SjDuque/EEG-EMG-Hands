from pylsl import StreamInfo, StreamOutlet
import time

# Create a new stream
info = StreamInfo('TestStream', 'EEG', 1, 100, 'float32', 'test_id')
outlet = StreamOutlet(info)

print("Stream is now active. Press Ctrl+C to stop.")
while True:
    outlet.push_sample([42.0])  # Sending dummy data
    time.sleep(0.01)
