#include <Arduino.h>
#include "BTS7960MotorDriver.h"

// --------------------------------------------------
// Configuration
// --------------------------------------------------
constexpr unsigned long BAUD_RATE = 115200; // use faster baud
constexpr int LED_PIN = 2;

// Motor driver instance (same pins as original setup)
BTS7960MotorDriverRC motorDriverAPI(
  32, 33, 21, 22,  // Motor 2 pins (RPWM2, LPWM2, REN2, LEN2)
  25, 26, 27, 14,  // Motor 1 pins (RPWM1, LPWM1, REN1, LEN1)
  34, 35, 12       // RC pins (unused here)
);

// Current target speeds for left/right motors
float targetSpeedLeft  = 0.0f;
float targetSpeedRight = 0.0f;

void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial) {}
  Serial.println("Drive endpoint ready. Listening for commands...");

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  motorDriverAPI.setup();
}

void loop() {
  // process all complete packets before updating motors
  bool updated = false;
  while (Serial.available() >= 4) {
    uint8_t packet[4];
    Serial.readBytes(packet, 4);

    if (packet[0] == 0x84) {
      uint8_t channel = packet[1];
      uint16_t rawTarget = (uint16_t)packet[2] | ((uint16_t)packet[3] << 7);

      float speed = ((float)rawTarget / 255.0f) * 2.0f - 1.0f;
      speed = constrain(speed, -1.0f, 1.0f);

      if (channel == 0) {
        targetSpeedLeft = speed;
        updated = true;
      } else if (channel == 1) {
        targetSpeedRight = speed;
        updated = true;
      }
    }
  }

  if (updated) {
    motorDriverAPI.setTargetSpeeds(targetSpeedLeft, targetSpeedRight);
  }
  motorDriverAPI.loop();
}
