import cv2
import numpy as np
from scipy import signal
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
from keras.models import Sequential
from keras.layers import Dense, Input
import serial
import sys
import termios
import tty

# Replace '/dev/tty.usbmodem2101' with the port your Arduino is connected to
arduino_port = '/dev/tty.usbmodem2101'

try:
    arduino = serial.Serial(arduino_port, 9600, timeout=1)
except AttributeError:
    print("Error: Ensure pyserial is installed and correctly imported.")
    sys.exit(1)

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def set_servo(servo_num, position):
    command = f"{servo_num} {position}\n"
    arduino.write(command.encode())

time.sleep(2)  # Wait for the connection to establish

# BrainFlow setup
params = BrainFlowInputParams()
params.serial_port = '/dev/cu.usbserial-D200PQ3N'
board_id = BoardIds.CYTON_BOARD.value

# Initialize the board
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()

# Constants
SCALE_FACTOR = (4500000) / 24 / (2 ** 23 - 1)
FS = 250  # Sampling rate
DATA_WINDOW_SIZE = 512  # Increased window size for better visualization
INTERVAL_DURATION = 7  # Interval duration in seconds
CLIP_START_DURATION = 1  # Duration to clip from the start of each interval
CLIP_END_DURATION = 1  # Duration to clip from the end of each interval

# Define channel configuration
channel_config = {
    0: 'emg',
}

# Initialize model variables
model = None
is_training = False
train_data = []
train_labels = []

# Initialize servo states
servo_states = {0: 180, 1: 0, 2: 0, 3: 0, 4: 0}  # Initial states of servos (0: open, 180: closed)
closed_state = {0: 180, 1: 0, 2: 0, 3: 0, 4: 0} # Closed state for all servos

# Set the initial states (open)
for servo_num in servo_states:
    set_servo(servo_num, 0)

def notch_filter(data, fs=250, notch_freq=60):
    bp_stop_hz = notch_freq + 3.0 * np.array([-1, 1])
    b, a = signal.butter(3, bp_stop_hz / (fs / 2.0), 'bandstop')
    return signal.lfilter(b, a, data)

def bandpass_filter(data, fs=250, lowcut=1, highcut=50):
    bp_hz = np.array([lowcut, highcut])
    b, a = signal.butter(5, bp_hz / (fs / 2.0), btype='bandpass')
    return signal.lfilter(b, a, data)

def create_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Single neuron with sigmoid activation for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_data, train_labels):
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    model.fit(train_data, train_labels, epochs=10, batch_size=32)
    return model

def main():
    global is_training, train_data, train_labels, model
    current_label = 1  # Start with flexing
    last_switch_time = time.time()
    last_prediction_time = time.time()
    predicted_label = 0
    prediction_color = (0, 0, 255)  # Default to red
    training_start_time = None
    collected_data_points = 0  # Track the number of collected data points

    emg_channels = BoardShim.get_emg_channels(board_id)  # Get EMG channels
    
    # Initialize model
    model = create_model(DATA_WINDOW_SIZE)

    # Initialize a buffer to store the waveform data
    waveform_buffer = np.zeros(DATA_WINDOW_SIZE)

    # For averaging predictions
    prediction_buffer = []

    # Update interval in seconds
    UPDATE_INTERVAL = 0.33

    while True:
        # Get data
        data = board.get_current_board_data(DATA_WINDOW_SIZE)
        emg_data = data[emg_channels[0]] * SCALE_FACTOR  # Use correct EMG channel index

        if len(emg_data) < DATA_WINDOW_SIZE:
            continue  # Ensure we have enough data points

        # Filter data
        emg_data = notch_filter(emg_data)
        emg_data = bandpass_filter(emg_data)

        # Update the waveform buffer
        waveform_buffer = np.roll(waveform_buffer, -len(emg_data))
        waveform_buffer[-len(emg_data):] = emg_data

        # Create a black image
        window = np.zeros((500, 800, 3), dtype=np.uint8)

        # Draw the waveform
        for i in range(1, DATA_WINDOW_SIZE):
            x1 = int((i - 1) * 800 / DATA_WINDOW_SIZE)
            y1 = int(250 - waveform_buffer[i - 1] * 100)
            x2 = int(i * 800 / DATA_WINDOW_SIZE)
            y2 = int(250 - waveform_buffer[i] * 100)
            cv2.line(window, (x1, y1), (x2, y2), (255, 255, 255), 1)

        if is_training:
            # Training phase
            current_time = time.time()
            elapsed_time = current_time - last_switch_time
            if training_start_time is None:
                training_start_time = current_time

            if elapsed_time > INTERVAL_DURATION:
                if current_label == 1:  # If currently flexing, switch to no flexing
                    current_label = 0
                    last_switch_time = current_time
                    elapsed_time = 0  # Reset elapsed time for the new interval
                else:  # If currently no flexing, end the training phase
                    is_training = False
                    model = train_model(model, train_data, train_labels)
                    train_data, train_labels = [], []  # Properly reset the lists
                    training_start_time = None
                    collected_data_points = 0

            # Skip the first second and the last second of data collection for each interval
            if CLIP_START_DURATION < elapsed_time < INTERVAL_DURATION - CLIP_END_DURATION:
                train_data.append(emg_data)
                train_labels.append(current_label)
                collected_data_points += 1

            color = (0, 255, 0) if current_label == 1 else (0, 0, 255)
            cv2.circle(window, (750, 50), 20, color, -1)

            # Indicate training mode
            cv2.putText(window, 'Training', (650, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Display countdown timer inside the circle
            remaining_time = INTERVAL_DURATION - int(elapsed_time)
            cv2.putText(window, str(remaining_time), (740, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        else:
            if model:
                # Update prediction every UPDATE_INTERVAL seconds
                prediction_buffer.append(model.predict(np.expand_dims(emg_data, axis=0)))
                if time.time() - last_prediction_time > UPDATE_INTERVAL:
                    average_prediction = np.mean(prediction_buffer, axis=0)
                    predicted_label = 1 if average_prediction >= 0.5 else 0
                    prediction_color = (0, 255, 0) if predicted_label == 1 else (0, 0, 255)
                    last_prediction_time = time.time()
                    prediction_buffer = []

                    # Control servos based on prediction
                    if predicted_label == 1:  # Focused (flex)
                        for servo_num in servo_states:
                            set_servo(servo_num, closed_state[servo_num])  # Close all fingers
                    else:  # Not focused (no flex)
                        for servo_num in servo_states:
                            set_servo(servo_num, 180-closed_state[servo_num])  # Open all fingers

                cv2.circle(window, (750, 50), 20, prediction_color, -1)

        cv2.imshow('EMG Data', window)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            is_training = True
            last_switch_time = time.time()
            current_label = 1  # Start with flexing
            train_data = []
            train_labels = []
            training_start_time = None
            collected_data_points = 0

    # Clean up
    board.stop_stream()
    board.release_session()
    arduino.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
