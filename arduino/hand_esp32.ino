#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include "BluetoothSerial.h"

// Create the PCA9685 object
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
BluetoothSerial SerialBT;

#define SERVOMIN  150  // This is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX  600  // This is the 'maximum' pulse length count (out of 4096)

// Function to convert degree angle to pulse length
uint16_t degreeToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32_Robotic_Hand");  // Bluetooth device name
  Wire.begin();  // Initialize I2C communication

  pwm.begin();
  pwm.setPWMFreq(60);  // Analog servos run at ~60 Hz update

  // Initialize all servos to 0 degrees
  for (uint8_t i = 0; i < 16; i++) {
    pwm.setPWM(i, 0, degreeToPulse(0));
  }

  Serial.println("ESP32 Ready");
  SerialBT.println("ESP32 Ready");
}

void loop() {
  if (SerialBT.available() > 0) {
    String input = SerialBT.readStringUntil('\n');
    input.trim();
    
    Serial.print("Received: ");
    Serial.println(input);
    SerialBT.print("Received: ");
    SerialBT.println(input);

    int spaceIndex = input.indexOf(' ');
    if (spaceIndex != -1) {
      String servoStr = input.substring(0, spaceIndex);
      String posStr = input.substring(spaceIndex + 1);
      
      int servoNum = servoStr.toInt();
      int position = posStr.toInt();
      
      Serial.print("Servo: ");
      Serial.print(servoNum);
      Serial.print(" Position: ");
      Serial.println(position);
      
      if (servoNum >= 0 && servoNum < 16 && position >= 0 && position <= 180) {
        pwm.setPWM(servoNum, 0, degreeToPulse(position));
      } else {
        Serial.println("Invalid servo number or position");
      }
    } else {
      Serial.println("Invalid command format");
    }
  }
}
