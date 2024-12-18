#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Create an instance of the PCA9685 using the default I2C address (0x40)
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Serial communication speed
const unsigned long SERIAL_BAUD_RATE = 115200;

// Define servo pulse length limits (adjust based on your servo specifications)
const int SERVO_MIN = 150;    // Minimum pulse length out of 4096
const int SERVO_MAX = 600;    // Maximum pulse length out of 4096

// Number of servos
const int NUM_SERVOS = 5;

// Servo states: Initialize with default positions (0 or 180 degrees)
const int default_states[NUM_SERVOS] = {180, 0, 0, 0, 0};
byte servo_angles[NUM_SERVOS];

// Start byte for synchronization
const byte START_BYTE = 0xAA;

// Function to map degree to PWM pulse
int degreeToPulse(int degree) {
  // Map 0-180 degrees to SERVO_MIN-SERVO_MAX pulse lengths
  return map(degree, 0, 180, SERVO_MIN, SERVO_MAX);
}

void setup() {
  // Initialize serial communication
  Serial.begin(SERIAL_BAUD_RATE);
  while (!Serial) {
    ; // Wait for serial port to connect. Needed for native USB
  }
  Serial.println("ESP32 Initialized. Waiting for commands...");

  // Initialize PCA9685
  pwm.begin();
  pwm.setPWMFreq(60); // Analog servos run at ~60 Hz

  // Initialize all servos to their initial states
  for (int i = 0; i < NUM_SERVOS; i++) {
    setServo(i, default_states[i]);
  }
}

void loop() {
  // Look for the start byte
  if (Serial.available() > 0) {
    byte incomingByte = Serial.read();
    if (incomingByte == START_BYTE) {
      // Check if the required number of bytes are available
      if (Serial.available() >= NUM_SERVOS) {
        Serial.readBytes(servo_angles, NUM_SERVOS);

        // Update servo states and set servos
        for (int i = 0; i < NUM_SERVOS; i++) {
          int angle = servo_angles[i];
          angle = abs(default_states[i] - angle);
          // Constrain angle to 0-180
          angle = constrain(angle, 0, 180);
          setServo(i, angle);
        }
      }
    }
  }
}

// Function to set servo position
void setServo(int servo_num, int degree) {
  int pulse = degreeToPulse(degree);
  pwm.setPWM(servo_num, 0, pulse);
}