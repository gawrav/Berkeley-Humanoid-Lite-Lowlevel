#include <Wire.h>
#include <Arduino.h>
#include <Adafruit_BNO08x.h>

void setup() {
  Serial.begin(115200);  // Use standard baud rate for testing
}

void loop() {
  Serial.println("hello");
  delay(500);
}

