import serial
import sys
from FingerAngles import FingerAngles  # Importing the HandTracking class

# Replace with your ESP32 Bluetooth serial port
esp32_port = '/dev/cu.ESP32_Hand'

try:
    esp32 = serial.Serial(esp32_port, 115200, timeout=1)
except serial.SerialException:
    print("Error: Ensure pyserial is installed and the port is correct.")
    sys.exit(1)

# Function to send servo commands to the ESP32
def set_servo(servo_num, position):
    position = max(0, min(180, position))  # Ensure the position is within valid range
    command = f"{servo_num} {position}\n"
    esp32.write(command.encode())
    # print(f"Sent: {command.strip()}")

def main():
    hand_tracker = FingerAngles()
    hand_tracker.start()

    try:
        while hand_tracker.isOpened():
            hand_tracker.update()
            
            # Retrieve angles and send commands to the ESP32
            for finger in hand_tracker.joint_sets.keys():
                angle = hand_tracker.get_angle(finger)
                servo_angle = int(hand_tracker.angle_to_percentage(finger, angle) * 1.8)  # Convert percentage to angle (0-180)
                
                # Round server angle to nearest 5 degrees
                servo_angle = round(servo_angle / 5) * 5
                
                # Send the command to the appropriate servo (example mapping, adjust as needed)
                if finger == 'thumb':
                    set_servo(0, 180-servo_angle)
                elif finger == 'index':
                    set_servo(1, servo_angle)
                elif finger == 'middle':
                    set_servo(2, servo_angle)
                elif finger == 'ring':
                    set_servo(3, servo_angle)
                elif finger == 'pinky':
                    set_servo(4, servo_angle)
    finally:
        hand_tracker.close()
        esp32.close()

if __name__ == "__main__":
    main()
