#include "DHT.h"

#define SOIL1_PIN 32
#define SOIL2_PIN 33
#define WATER_PIN 34
#define RAIN_PIN 27

#define DHT_PIN 4
#define DHT_TYPE DHT22

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);

  pinMode(RAIN_PIN, INPUT);

  dht.begin();

  delay(2000);

  Serial.println();
  Serial.println("====================================");
  Serial.println("   SMART AGRICULTURE SENSOR SYSTEM");
  Serial.println("====================================");
}

void loop() {

  int soil1 = analogRead(SOIL1_PIN);
  int soil2 = analogRead(SOIL2_PIN);
  int water = analogRead(WATER_PIN);
  int rain = digitalRead(RAIN_PIN);

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  Serial.println();
  Serial.println("------------------------------------");

  Serial.print("Soil Moisture 1 : ");
  Serial.println(soil1);

  Serial.print("Soil Moisture 2 : ");
  Serial.println(soil2);

  Serial.print("Temperature     : ");
  if (isnan(temperature))
    Serial.println("Sensor Error");
  else {
    Serial.print(temperature);
    Serial.println(" °C");
  }

  Serial.print("Humidity        : ");
  if (isnan(humidity))
    Serial.println("Sensor Error");
  else {
    Serial.print(humidity);
    Serial.println(" %");
  }

  Serial.print("Water Level     : ");
  Serial.println(water);

  Serial.print("Rain Status     : ");
  if (rain == 0)
    Serial.println("RAIN DETECTED");
  else
    Serial.println("NO RAIN");

  Serial.println("------------------------------------");

  delay(2000);
}