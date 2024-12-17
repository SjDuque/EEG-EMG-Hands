import serial
import time
import sys
import termios
import tty

# Replace with the correct port for your ESP32
esp32_port = '/dev/cu.usbserial-110'  # Update this to match your system

def initialize_serial(port):
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)  # Wait for the connection to establish
        print("Serial connection established.")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

def close_serial(ser):
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")

def getch():
    """
    Capture a single character from the keyboard without echoing to the screen.
    This function works on Unix-like systems.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def send_servo_angles(ser, angles):
    """
    Send servo angles as 5 bytes to the ESP32.
    Each angle must be an integer between 0 and 180.
    """
    try:
        # Ensure all angles are within 0-180
        bytes_to_send = bytes([angle if 0 <= angle <= 180 else 0 for angle in angles])
        ser.write(bytes_to_send)
        print(f"Sent angles: {angles}")
        time.sleep(0.01)  # Short delay to ensure the command is processed
    except serial.SerialException as e:
        print(f"Error sending angles: {e}")

def main():
    esp32 = initialize_serial(esp32_port)

    if not esp32:
        print("Failed to connect to ESP32. Retrying...")
        while not esp32:
            time.sleep(2)
            esp32 = initialize_serial(esp32_port)

    print("Press 'A S D F SPACE' to toggle the fingers.")
    print("Press 'q' to quit.")

    # Mapping keys to servo indices
    finger_map = {
        'a': 4,
        's': 3,
        'd': 2,
        'f': 1,
        ' ': 0,  # Assuming 'g' is mapped to servo 0
    }

    # Initialize servo states
    servo_states = [180, 0, 0, 0, 0]  # servo 0 to 4

    # Set the initial states
    send_servo_angles(esp32, servo_states)

    try:
        while True:
            key = getch().lower()
            if key == 'q':
                break
            elif key in finger_map:
                servo_num = finger_map[key]
                # Toggle the state
                if servo_states[servo_num] == 0:
                    servo_states[servo_num] = 180
                else:
                    servo_states[servo_num] = 0
                send_servo_angles(esp32, servo_states)
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
    finally:
        close_serial(esp32)

if __name__ == "__main__":
    main()
