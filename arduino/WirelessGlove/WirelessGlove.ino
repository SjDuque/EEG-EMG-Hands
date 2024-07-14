#include <SoftwareSerial.h>

#define BTH_RX 11
#define BTH_TX 12
#define K3_BUTTON_PIN 7

SoftwareSerial Bth(BTH_RX, BTH_TX);

void setup() {
  Serial.begin(9600);
  Bth.begin(9600);
  pinMode(K3_BUTTON_PIN, INPUT_PULLUP);

  Bth.print("AT+ROLE=0");  // Set HC-08 to slave mode
  delay(100);
  Bth.print("AT+RESET");  // Reset the HC-08 module
  delay(250);

  Serial.println("Setup complete. Waiting for K3 button press...");
}

void loop() {
  if (digitalRead(K3_BUTTON_PIN) == LOW) {
    Serial.println("K3 button pressed. Sending message...");
    Bth.println("sending");
    delay(1000);  // Debounce delay
  }
  
  if (Bth.available()) {
    String receivedMessage = Bth.readString();
    Serial.print("Received from ESP32: ");
    Serial.println(receivedMessage);
  }
}
