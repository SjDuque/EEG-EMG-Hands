import serial
import sys
import time
from typing import List, Tuple, Union

class HandSerial:
    """
    _summary_: Class to send finger percentages to the ESP32 via serial.
    """
    
    def __init__(self, serial_port:str, baud_rate:int=115200, timeout:int=1, left_hand:bool=False, right_hand:bool=False):
        """
        _summary_: Initialize the HandSerial class.

        Args:
            serial_port (str): _description_
            baud_rate (int, optional): _description_. Defaults to 115200.
            timeout (int, optional): _description_. Defaults to 1.
            left_hand (bool, optional): _description_. Defaults to True.
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None
        self.start_byte = 0xAA
        self.servo_min = 0
        self.servo_max = 180
        self.num_servos = 5
        self.prev_servo_angles = [0.5] * self.num_servos
        self.connect()
        
        if left_hand == right_hand or right_hand:
            self.right_hand = True
        else:
            self.right_hand = False
        
    def connect(self):
        """
        _summary_: Connect to the ESP32 via serial.
        """
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=self.timeout)
            time.sleep(self.timeout + 0.5)  # Wait for serial connection to initialize
            print(f"Connected to ESP32 on {self.serial_port}")
        except serial.SerialException as e:
            print(f"Failed to connect to ESP32 on {self.serial_port}: {e}")
            sys.exit(1)

    def send_serial(self, finger_percentages: Union[List[float], Tuple[float]]):
        """
        _summary_: Send finger percentages to the ESP32 via serial.
        """
        if self.ser is None:
            print("Serial connection is not initialized.")
            return
        
        if finger_percentages is None or len(finger_percentages) != self.num_servos:
            return
        
        # Convert finger percentages to servo angles
        servo_angles = [int(self.servo_min + (self.servo_max - self.servo_min) * p) for p in finger_percentages]
        servo_angles = [max(self.servo_min, min(self.servo_max, a)) for a in servo_angles]
        
        # Right hand: Reverse the order of servo angles
        if self.right_hand:
            servo_angles = servo_angles[::-1]
        
        # Check if servo angles are the same as previous
        if servo_angles == self.prev_servo_angles:
            return
        
        # Send start byte and servo angles
        self.ser.write(bytes([self.start_byte] + servo_angles))
        
        # Update previous servo angles
        self.prev_servo_angles = servo_angles
        
    def close(self):
        if self.ser is not None:
            self.ser.close()
            print("Serial connection closed.")