import pylsl
import numpy as np
import serial
import serial.tools.list_ports
import time
import sys

def find_serial_port(baudrate=115200, timeout=1):
    """
    Attempts to find the serial port connected to the ESP32.
    Adjust this function based on your system's serial port naming conventions.
    """
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        # You can add more sophisticated checks here based on port.description or port.manufacturer
        if "ESP32" in port.description or "USB" in port.description:
            try:
                ser = serial.Serial(port.device, baudrate, timeout=timeout)
                ser.close()
                return port.device
            except (OSError, serial.SerialException):
                continue
    return None

def mediapipe_client_send_serial():
    # ------------------ Configuration ------------------
    SERIAL_BAUD_RATE = 115200
    SERIAL_TIMEOUT = 1  # in seconds
    SERIAL_PORT = None  # Set to None to auto-detect

    LSL_STATUS_NAME = "FingerStatus"

    NUM_SERVOS = 5  # Thumb, Index, Middle, Ring, Pinky
    SERVO_MIN = 0
    SERVO_MAX = 180
    
    LEFT_HAND = True
    # ---------------------------------------------------

    # Initialize serial connection
    if SERIAL_PORT is None:
        SERIAL_PORT = find_serial_port(baudrate=SERIAL_BAUD_RATE, timeout=SERIAL_TIMEOUT)
        if SERIAL_PORT is None:
            print("Could not find ESP32 serial port. Please specify it manually in the script.")
            sys.exit(1)
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD_RATE, timeout=SERIAL_TIMEOUT)
        time.sleep(2)  # Wait for serial connection to initialize
        print(f"Connected to ESP32 on {SERIAL_PORT}")
    except serial.SerialException as e:
        print(f"Failed to connect to ESP32 on {SERIAL_PORT}: {e}")
        sys.exit(1)

    # Resolve LSL streams
    print("Looking for HandLandmarks stream...")
    streams = pylsl.resolve_byprop("name", LSL_STATUS_NAME)
    if not streams:
        print(f"No {LSL_STATUS_NAME} stream found.")
        return

    inlet_status= pylsl.StreamInlet(streams[0], processing_flags=pylsl.proc_ALL)

    # Define finger indices for percentages (assuming order: thumb, index, middle, ring, pinky)
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    # Initialize previous servo angles to detect changes
    prev_servo_angles = [None] * NUM_SERVOS

    try:
        while True:
            # Pull latest samples
            status_samples, _ = inlet_status.pull_chunk(timeout=0.01)

            if not status_samples:
                time.sleep(0.1)
                continue
            
            status_samples = status_samples[-1]  # Use the latest sample
            for i in range(len(status_samples)):

                servo_angles = []
                for i in range(NUM_SERVOS):
                    if LEFT_HAND:
                        a = int(status_samples[NUM_SERVOS-1-i] * (SERVO_MAX - SERVO_MIN) + SERVO_MIN)
                    else:
                        a = int(status_samples[NUM_SERVOS-1-i] * (SERVO_MAX - SERVO_MIN) + SERVO_MIN)
                    servo_angles.append(a)

                # Send angles via serial if they have changed
                if servo_angles != prev_servo_angles:
                    try:
                        ser.write(bytes(servo_angles))
                        prev_servo_angles = servo_angles.copy()
                        # Optional: Print sent angles
                        print(f"Sent angles: {servo_angles}")
                    except serial.SerialException as e:
                        print(f"Serial communication error: {e}")
                        break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        ser.close()

if __name__ == "__main__":
    mediapipe_client_send_serial()
