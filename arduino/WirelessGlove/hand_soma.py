import serial
import time

# Replace with the correct port for your Arduino and ESP32
arduino_port = '/dev/cu.wchusbserial10'  # Update with your Arduino port
esp32_port = '/dev/cu.ESP32_Robotic_Hand'  # Update with your ESP32 Bluetooth port
POT_ROUND = 1  # Round the potentiometer values to the nearest n value

def initialize_serial(port, baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for the connection to establish
        print(f"Serial connection established on {port}.")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

def close_serial(ser):
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")

def is_valid_data(data):
    try:
        # Split the data and try to convert each part to a float
        values = data.split(',')
        _ = [float(value) for value in values]
        return True
    except ValueError:
        return False

def read_potentiometers(ser):
    try:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8', errors='ignore').strip()
            if data and is_valid_data(data):  # Ensure the data is valid
                pot_values = list(map(float, data.split(',')))
                return pot_values
    except serial.SerialException as e:
        print(f"Error reading from Arduino: {e}")
    return None

def map_value(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def set_servo(ser, servo_num, position):
    command = f"{servo_num} {position}\n"
    try:
        ser.write(command.encode())
        print(f"Sent: {command.strip()}")
        # time.sleep(0.05)  # Short delay to ensure the command is processed
    except serial.SerialException as e:
        print(f"Error sending command: {e}")

def main():
    arduino = initialize_serial(arduino_port, 9600)
    esp32 = initialize_serial(esp32_port, 115200)

    if not arduino or not esp32:
        print("Failed to connect to Arduino or ESP32.")
        return

    try:
        while True:
            pot_values = read_potentiometers(arduino)
            if pot_values:
                print(f"Potentiometer values: {pot_values}")
                for i, value in enumerate(pot_values):
                    position = map_value(value, 500, 2500, 0, 180)  # Map the potentiometer value to servo angle
                    if i == 0:  # Reverse the thumb servo direction
                        position = 180 - position
                    position = round(position / POT_ROUND) * POT_ROUND  # Round to the nearest 2
                    set_servo(esp32, i, position)
            
            # time.sleep(0.05)  # Adjust the delay as needed

    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
    finally:
        close_serial(arduino)
        close_serial(esp32)

if __name__ == "__main__":
    main()
