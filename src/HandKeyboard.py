import serial
import time
import sys
import termios
import tty

# Replace with the correct port for your ESP32
esp32_port = '/dev/cu.ESP32_Hand'

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
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def set_servo(ser, servo_num, position):
    command = f"{servo_num} {position}\n"
    try:
        ser.write(command.encode())
        print(f"Sent: {command.strip()}")
        time.sleep(0.1)  # Short delay to ensure the command is processed
    except serial.SerialException as e:
        print(f"Error sending command: {e}")

def main():
    esp32 = initialize_serial(esp32_port)

    if not esp32:
        print("Failed to connect to ESP32. Retrying...")
        while not esp32:
            time.sleep(2)
            esp32 = initialize_serial(esp32_port)

    print("Press 'ASDFG' to toggle the fingers.")
    print("Press 'q' to quit.")

    finger_map = {
        'a': 4,
        's': 3,
        'd': 2,
        'f': 1,
        ' ': 0,
    }

    servo_states = {0: 180, 1: 0, 2: 0, 3: 0, 4: 0}  # Initial states of servos (0: open, 180: closed)

    # Set the initial states
    for servo_num, state in servo_states.items():
        set_servo(esp32, servo_num, state)

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
                set_servo(esp32, servo_num, servo_states[servo_num])
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
    finally:
        close_serial(esp32)

if __name__ == "__main__":
    main()
