import pylsl
import numpy as np
import cv2
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

    LSL_HAND_LANDMARKS_NAME = "HandLandmarks"
    LSL_FINGER_PERCENTAGES_NAME = "FingerPercentages"

    NUM_SERVOS = 5  # Thumb, Index, Middle, Ring, Pinky
    SERVO_MIN = 0
    SERVO_MAX = 180

    img_size = 360
    scale = img_size  # to scale x,y from [0,1] to image coordinates
    font_scale = 0.45
    font_thickness = 2

    # Define connections between landmarks for visualization
    connections = {
        "palm":     ((0, 165, 255),  [0, 1, 5, 9, 13, 17, 0]),
        "thumb":    ((0, 255, 0),    [1, 2, 3, 4]),
        "index":    ((0, 0, 255),    [5, 6, 7, 8]),
        "middle":   ((255, 0, 0),    [9, 10, 11, 12]),
        "ring":     ((0, 255, 255),  [13, 14, 15, 16]),
        "pinky":    ((255, 0, 255),  [17, 18, 19, 20]),
    }
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
    streams_landmarks = pylsl.resolve_byprop("name", LSL_HAND_LANDMARKS_NAME)
    if not streams_landmarks:
        print(f"No {LSL_HAND_LANDMARKS_NAME} stream found.")
        return

    print("Looking for FingerPercentages stream...")
    streams_percentages = pylsl.resolve_byprop("name", LSL_FINGER_PERCENTAGES_NAME)
    if not streams_percentages:
        print(f"No {LSL_FINGER_PERCENTAGES_NAME} stream found.")
        return

    inlet_landmarks = pylsl.StreamInlet(streams_landmarks[0], processing_flags=pylsl.proc_ALL)
    inlet_percentages = pylsl.StreamInlet(streams_percentages[0], processing_flags=pylsl.proc_ALL)

    # Print inlet info (optional)
    print("Landmarks Inlet Info:")
    print(inlet_landmarks.info().as_xml())
    print("Finger Percentages Inlet Info:")
    print(inlet_percentages.info().as_xml())

    # Assume 21 landmarks, each with x,y,z
    num_landmarks = 21

    # Define finger indices for percentages (assuming order: thumb, index, middle, ring, pinky)
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_thresholds = ((0.75, 0.9), (0.5, 0.85), (0.5, 0.85), (0.3, 0.85), (0.3, 0.85))

    # Initialize previous servo angles to detect changes
    prev_servo_angles = [None] * NUM_SERVOS

    try:
        while True:
            # Pull latest samples
            landmark_samples, _ = inlet_landmarks.pull_chunk(timeout=0.01)
            percentage_samples, _ = inlet_percentages.pull_chunk(timeout=0.01)

            if landmark_samples and percentage_samples:
                landmark_sample = landmark_samples[-1]
                percentage_sample = percentage_samples[-1]
                for i in range(len(percentage_sample)):
                    if np.isnan(percentage_sample[i]):
                        percentage_sample[i] = 0.5
                    else:
                        percentage_sample[i] = (percentage_sample[i] - finger_thresholds[i][0]) / (finger_thresholds[i][1] - finger_thresholds[i][0])
                        percentage_sample[i] = np.clip(percentage_sample[i], 0, 1)
                        
            else:
                landmark_sample = None
                percentage_sample = None

            if (landmark_sample is not None and len(landmark_sample) == num_landmarks * 3) and (percentage_sample is not None and len(percentage_sample) >= NUM_SERVOS):
                # Map finger percentages to servo angles
                servo_angles = []
                for i in range(NUM_SERVOS):
                    a = int(percentage_sample[i] * (SERVO_MAX - SERVO_MIN) + SERVO_MIN)
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
                    
                # Convert landmarks to (21, 3)
                coords = np.array(landmark_sample).reshape(num_landmarks, 3)

                # Optional: Visualization
                img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

                if not np.isnan(coords).any():
                    # Draw connections
                    for connection, (color, landmark_list) in connections.items():
                        thickness = 2
                        for i in range(len(landmark_list)-1):
                            pt1 = tuple((coords[landmark_list[i]][:2] * scale).astype(int))
                            pt2 = tuple((coords[landmark_list[i+1]][:2] * scale).astype(int))
                            cv2.line(img, pt1, pt2, color, thickness)

                    # Draw landmarks
                    for idx, (x, y, z) in enumerate(coords):
                        center = tuple((np.array([x, y]) * scale).astype(int))
                        cv2.circle(img, center, 3, (255, 255, 255), -1)

                    # Optionally, display finger percentages
                    finger_texts = []
                    percentage_texts = []
                    for i, a in enumerate(percentage_sample[:NUM_SERVOS]):
                        finger_texts.append(finger_names[i])
                        if np.isnan(a):
                            percentage_texts.append(f"NaN")
                        else:
                            percentage = int(a * 100)
                            percentage_texts.append(f"{percentage}%")
                            
                    
                    finger_text = " ".join(finger_texts)
                    percentage_text = " ".join(percentage_texts)
                    cv2.putText(img, finger_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                    cv2.putText(img, percentage_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

                cv2.imshow("Hand Tracking Client", img)

            # Handle OpenCV window events
            if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit
                print("Exiting...")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mediapipe_client_send_serial()
