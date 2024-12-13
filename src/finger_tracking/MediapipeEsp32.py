import serial
import time
import threading
from finger_tracking.FingerAngles import FingerAngles

# Serial settings
SERIAL_PORT = "/dev/cu.wchusbserial130"  # Replace with your ESP32's serial port
BAUD_RATE = 115200
SEND_RATE = 60  # Hz (60 packets per second)
SEND_INTERVAL = 1.0 / SEND_RATE  # Interval between sends in seconds

# Initialize tick counter (0-255)
current_tick = 0
tick_lock = threading.Lock()

# Dictionary to store sent ticks and their send times
sent_ticks = {}
sent_ticks_lock = threading.Lock()

# Flag to control the running state of threads
running = True

def send_packets(ser, hand_tracker):
    global current_tick, running
    while running and hand_tracker.isOpened():
        start_time = time.time()
        hand_tracker.update(draw_camera=True, terminal_out=False)  # Disable camera display and terminal output for faster processing
        
        if hand_tracker.found_hand:
            # Retrieve angles and prepare servo data
            servo_data = []
            for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                angle = hand_tracker.get_angle(finger)
                percentage = hand_tracker.angle_to_percentage(finger, angle)
                m = 5
                percentage = int(percentage / m) * m / 100
                
                print(f"{finger}: {angle:.2f} degrees, {percentage:.2f}%")

                # Clamp the percentage between 0 and 1
                percentage = max(0.0, min(1.0, percentage))

                # Convert percentage to byte (0-255)
                servo_byte = int(round(percentage * 255))
                
                # # Round to nearest m multiple
                # m = 5
                # servo_byte = m * round(servo_byte / m)

                # Invert thumb if necessary
                if finger == 'thumb':
                    servo_byte = 255 - servo_byte

                servo_data.append(servo_byte)

            # Ensure exactly 5 bytes
            if len(servo_data) == 5:
                with tick_lock:
                    tick = current_tick
                    current_tick = (current_tick + 1) % 256  # Wrap around at 256

                # Append tick to servo data
                packet = bytes(servo_data + [tick])

                # Record the send time
                with sent_ticks_lock:
                    sent_ticks[tick] = time.time()

                # Send the packet
                try:
                    ser.write(packet)
                except Exception as e:
                    print(f"Error sending packet: {e}")

                # Optionally, print sent data for debugging
                # print(f"Sent tick {tick}: {list(servo_data)}")
        
        # Sleep to maintain the send rate
        elapsed = time.time() - start_time
        sleep_time = SEND_INTERVAL - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # If processing is taking too long, skip sleeping
            pass

def receive_acks(ser):
    global running
    while running:
        try:
            ack = ser.read()  # Read 1 byte for the ACK tick
            if ack:
                ack_tick = ack[0]

                with sent_ticks_lock:
                    if ack_tick in sent_ticks:
                        send_time = sent_ticks.pop(ack_tick)
                        recv_time = time.time()
                        latency = (recv_time - send_time) * 1000  # Convert to milliseconds
                        print(f"Tick {ack_tick}: Latency = {latency:.2f} ms")
                    else:
                        print(f"Received ACK with unknown or expired tick: {ack_tick}")
        except Exception as e:
            print(f"Error receiving ACK: {e}")

def main():
    global running
    # Initialize serial connection
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")
    except Exception as e:
        print(f"Error: Could not open serial port {SERIAL_PORT}: {e}")
        return

    # Initialize hand tracker
    hand_tracker = FingerAngles()
    hand_tracker.start()

    # Start the ACK receiver thread
    receiver_thread = threading.Thread(target=receive_acks, args=(ser,), daemon=True)
    receiver_thread.start()

    # Start sending packets
    try:
        send_packets(ser, hand_tracker)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Signal threads to stop
        running = False
        hand_tracker.close()
        ser.close()
        receiver_thread.join()

if __name__ == "__main__":
    main()
