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
int servo_states[NUM_SERVOS] = {180, 0, 0, 0, 0}; // Servos 0 to 4

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
    setServo(i, servo_states[i]);
  }
}

void loop() {
  // Check if enough bytes are available (5 bytes)
  if (Serial.available() >= NUM_SERVOS) {
    byte servo_angles[NUM_SERVOS];
    Serial.readBytes(servo_angles, NUM_SERVOS);

    // Update servo states and set servos
    for (int i = 0; i < NUM_SERVOS; i++) {
      int angle = servo_angles[i];
      // Constrain angle to 0-180
      angle = constrain(angle, 0, 180);
      servo_states[i] = angle;
      setServo(i, angle);
    }

    // Optional: Send back confirmation
    Serial.print("Servos updated: ");
    for (int i = 0; i < NUM_SERVOS; i++) {
      Serial.print("Servo ");
      Serial.print(i);
      Serial.print(" = ");
      Serial.print(servo_states[i]);
      if (i < NUM_SERVOS - 1) Serial.print(", ");
    }
    Serial.println();
  }
}

// Function to set servo position
void setServo(int servo_num, int degree) {
  int pulse = degreeToPulse(degree);
  pwm.setPWM(servo_num, 0, pulse);
}
