// Define your analog pin
const int sensorPin = 34; // Make sure this pin is connected to your sensor

// Sampling frequency control
volatile unsigned long samplingInterval = 1000; // Default: 1Hz (1000 ms)
unsigned long lastSampleTime = 0;

// Sensor data
int currentSample = 0;
int highestSample = 0;
int lowestSample = 3000;

// Peak detection
int highCandidate = 0;
int highCount = 0;
int lowCandidate = 4095;
int lowCount = 0;
const int thresholdCount = 2; // Require 10 confirmations

// Serial input buffer
String inputString = "";
bool stringComplete = false;


float adcToVoltage(int x) {
  return 3.99e-9 * pow(x, 4) -
         2.30e-5 * pow(x, 3) +
         0.0449  * pow(x, 2) -
         32.41   * x +
         5125.0;
}






void setup() {
  Serial.begin(115200);               // Set baud rate
  inputString.reserve(10);            // Reserve buffer for frequency input
  delay(2000);                         // Wait for Serial Monitor to start
  Serial.println("Ready. Enter frequency in Hz (e.g. 10):");
}

void loop() {
  // Handle user input from serial monitor
  if (stringComplete) {
    int freq = inputString.toInt();
    if (freq > 0) {
      samplingInterval = 1000 / freq;
      Serial.print("Sampling frequency set to: ");
      Serial.print(freq);
      Serial.println(" Hz");
    } else {
      Serial.println("Invalid input. Please enter a positive integer.");
    }
    inputString = "";
    stringComplete = false;
  }

  // Sampling logic
  unsigned long now = millis();
  if (now - lastSampleTime >= samplingInterval) {
    lastSampleTime = now;
    currentSample = analogRead(sensorPin);

    // High value tracking
    if (currentSample > highestSample) {
      if (currentSample == highCandidate) {
        highCount++;
      } else {
        highCandidate = currentSample;
        highCount = 1;
      }
      if (highCount >= thresholdCount) {
        highestSample = highCandidate;
      }
    }

    // Low value tracking
    if (currentSample < lowestSample && currentSample > 500) {
      if (currentSample == lowCandidate) {
        lowCount++;
      } else {
        lowCandidate = currentSample;
        lowCount = 1;
      }
      if (lowCount >= thresholdCount) {
        lowestSample = lowCandidate;
      }
    }

    // ===== Plot-friendly Serial Output =====
    // Tab-separated values with no labels — required for Arduino Serial Plotter
    Serial.print(currentSample);
    Serial.print("\t");
    Serial.print(highestSample);
    Serial.print("\t");
    Serial.print(lowestSample);
    Serial.print("\t");

    float voltage = adcToVoltage(currentSample);
    float highestVoltage = adcToVoltage(highestSample);
    float lowestVoltage = adcToVoltage(lowestSample);

    Serial.print(voltage, 1);
    Serial.print("\t");
    Serial.print(highestVoltage, 1);
    Serial.print("\t");
    Serial.println(lowestVoltage, 1);

  }

}

// Read input asynchronously
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}

// 2389 228 v  lowest is 1265    

// ~1870 0 v 

//2355   1274
//322.4406922210657

// 2361 1264  
//353.5533905932738

