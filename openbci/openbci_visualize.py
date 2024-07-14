import cv2
import numpy as np
from scipy import signal
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Setup BrainFlow parameters
params = BrainFlowInputParams()
params.serial_port = '/dev/cu.usbserial-D30DLPKS'
board_id = BoardIds.CYTON_BOARD.value

# Initialize the board
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()

# Constants
SCALE_FACTOR_EMG = 1  # Reduced scale factor for better visibility
SCALE_FACTOR_EEG = (4500000) / 24 / (2 ** 23 - 1)  # EEG scale factor
FS = 250  # Sampling rate
DATA_WINDOW_SIZE = 512
CHANNEL_SPACING = 10
TOP_BOTTOM_MARGIN = 20

# Customizable dimensions for each channel
CHANNEL_WIDTH = 800
CHANNEL_HEIGHT = 100

# Debug flag
DEBUG = True

# Specify channels and their types (index, type)
channels_to_display = [(0, 'EMG'), (1, 'EMG'), (2, 'EMG'), (3, 'EMG'), (4, 'EMG'), (5, 'EMG'), (6, 'EMG'), (7, 'EMG')]

def notch_filter(data, fs=250, notch_freq=60):
    bp_stop_hz = notch_freq + 3.0 * np.array([-1, 1])
    b, a = signal.butter(3, bp_stop_hz / (fs / 2.0), 'bandstop')
    return signal.lfilter(b, a, data)

def bandpass_filter(data, fs=250, lowcut=20, highcut=100):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1:
        raise ValueError(f"Filter critical frequencies out of range: low={low}, high={high}")
    b, a = signal.butter(5, [low, high], btype='bandpass')
    return signal.lfilter(b, a, data)

def process_and_plot_channel(data, idx, window, y_offset, ch_type):
    if len(data) < DATA_WINDOW_SIZE:
        if DEBUG:
            print(f"Not enough data points for channel {idx+1}: {len(data)}")
        return

    # Filter data
    data = notch_filter(data)
    data = bandpass_filter(data)

    if ch_type == 'EMG':
        # Time-domain plot
        for i in range(len(data) - 1):
            x1 = int(i * (CHANNEL_WIDTH / DATA_WINDOW_SIZE))
            x2 = int((i + 1) * (CHANNEL_WIDTH / DATA_WINDOW_SIZE))
            y1 = int(y_offset - data[i] * SCALE_FACTOR_EMG)
            y2 = int(y_offset - data[i + 1] * SCALE_FACTOR_EMG)
            cv2.line(window, (x1, y1), (x2, y2), (0, 255, 0), 1)
    else:
        # Compute FFT
        sp = np.abs(np.fft.fft(data))
        freq = np.fft.fftfreq(len(data), 1.0 / FS)

        # Plot FFT on the image
        max_sp = max(sp)
        for i in range(len(freq)):
            if freq[i] > 0 and freq[i] < FS / 2:
                x = int(CHANNEL_WIDTH * (freq[i] / (FS / 2)))
                y = int(CHANNEL_HEIGHT * (sp[i] / max_sp))
                cv2.line(window, (x, y_offset), (x, y_offset - y), (255, 0, 0), 1)

def main():
    while True:
        # Get data
        data = board.get_current_board_data(DATA_WINDOW_SIZE)

        eeg_channels = BoardShim.get_eeg_channels(board_id)
        eeg_data = data[eeg_channels] * SCALE_FACTOR_EEG  # Get all EEG channels data
        
        emg_channels = BoardShim.get_emg_channels(board_id)
        emg_data = data[emg_channels] * SCALE_FACTOR_EMG  # Get all EMG channels data

        # Calculate window size based on the number of channels to display
        num_channels = len(channels_to_display)
        window_height = num_channels * (CHANNEL_HEIGHT + CHANNEL_SPACING) + 2 * TOP_BOTTOM_MARGIN

        # Create a black image
        window = np.zeros((window_height, CHANNEL_WIDTH, 3), dtype=np.uint8)

        # Sort channels to display by index
        sorted_channels = sorted(channels_to_display, key=lambda x: x[0])

        for idx, (channel, ch_type) in enumerate(sorted_channels):
            if ch_type == 'EMG':
                if channel < len(emg_data):
                    channel_data = emg_data[channel]
                else:
                    if DEBUG:
                        print(f"Channel {channel} is out of range for EMG data.")
                    continue
            else:
                if channel < len(eeg_data):
                    channel_data = eeg_data[channel]
                else:
                    if DEBUG:
                        print(f"Channel {channel} is out of range for EEG data.")
                    continue
                
            y_offset = (CHANNEL_HEIGHT + CHANNEL_SPACING) * (idx + 1) + TOP_BOTTOM_MARGIN
            if DEBUG:
                print(f"Processing channel {channel} ({ch_type})")
            process_and_plot_channel(channel_data, channel, window, y_offset, ch_type)

        # Show the image
        cv2.imshow('EEG/EMG Data', window)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    board.stop_stream()
    board.release_session()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
