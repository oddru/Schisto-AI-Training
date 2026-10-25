#include <Arduino.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

namespace {

constexpr uint8_t kGateOpenPin = 18;
constexpr uint8_t kGateClosedPin = 19;
constexpr uint8_t kSafeModePin = 21;
constexpr float kMinimumDepthCm = -20.0F;
constexpr float kMaximumDepthCm = 10.0F;
constexpr float kAwdMinimumDepthCm = -15.0F;
constexpr float kFloweringMinimumDepthCm = 0.0F;
constexpr float kMaximumDepthLimitCm = 5.0F;
constexpr float kOpenDepthChangeCmPerHour = 1.0F;
constexpr float kClosedDepthChangeCmPerHour = -0.5F;
constexpr long kMaximumReadingAgeMs = 3600000L;
constexpr size_t kLineCapacity = 128;
constexpr uint8_t kMaximumTokens = 7;

struct Reading {
  char stage[16];
  float depthCm;
  bool requestedOpen;
  long ageMs;
  bool faultFlag;
};

bool gateOpen = false;
bool safeMode = false;
bool havePreviousDepth = false;
float previousDepthCm = 0.0F;
uint8_t identicalDepthCount = 0;
char inputLine[kLineCapacity];
size_t inputLength = 0;
bool inputOverflow = false;

void updateIndicators() {
  digitalWrite(kGateOpenPin, gateOpen ? HIGH : LOW);
  digitalWrite(kGateClosedPin, gateOpen ? LOW : HIGH);
  digitalWrite(kSafeModePin, safeMode ? HIGH : LOW);
}

void enterSafeMode(const char *reason) {
  safeMode = true;
  updateIndicators();
  Serial.print("RESULT gate=");
  Serial.print(gateOpen ? "OPEN" : "CLOSED");
  Serial.print(" safe=1 reason=");
  Serial.println(reason);
}

bool parseFloatToken(const char *token, float &value) {
  if (token == nullptr || *token == '\0') {
    return false;
  }
  char *end = nullptr;
  value = strtof(token, &end);
  return end != token && *end == '\0';
}

bool parseLongToken(const char *token, long &value) {
  if (token == nullptr || *token == '\0') {
    return false;
  }
  char *end = nullptr;
  value = strtol(token, &end, 10);
  return end != token && *end == '\0';
}

bool parseBinaryToken(const char *token, bool &value) {
  long parsed = 0;
  if (!parseLongToken(token, parsed) || (parsed != 0 && parsed != 1)) {
    return false;
  }
  value = parsed == 1;
  return true;
}

bool parseReading(char **tokens, uint8_t count, Reading &reading) {
  if (count != 6) {
    return false;
  }
  if (strlen(tokens[1]) >= sizeof(reading.stage)) {
    return false;
  }
  strcpy(reading.stage, tokens[1]);
  return parseFloatToken(tokens[2], reading.depthCm) &&
         parseBinaryToken(tokens[3], reading.requestedOpen) &&
         parseLongToken(tokens[4], reading.ageMs) &&
         parseBinaryToken(tokens[5], reading.faultFlag);
}

bool isKnownStage(const char *stage) {
  return strcmp(stage, "TRANSPLANT") == 0 ||
         strcmp(stage, "VEG") == 0 ||
         strcmp(stage, "FLOWER") == 0 ||
         strcmp(stage, "RIPEN") == 0;
}

bool isFlowering(const char *stage) {
  return strcmp(stage, "FLOWER") == 0;
}

bool passesValidityChecks(const Reading &reading, const char *&reason) {
  if (reading.faultFlag) {
    reason = "sensor_fault_flag";
    return false;
  }
  if (reading.ageMs < 0 || reading.ageMs > kMaximumReadingAgeMs) {
    reason = "stale_reading";
    return false;
  }
  if (!isfinite(reading.depthCm)) {
    reason = "non_finite:depth_cm";
    return false;
  }
  if (reading.depthCm < kMinimumDepthCm ||
      reading.depthCm > kMaximumDepthCm) {
    reason = "out_of_range:depth_cm";
    return false;
  }
  if (!isKnownStage(reading.stage)) {
    reason = "invalid:crop_stage";
    return false;
  }
  return true;
}

bool isFrozenDepth(float depthCm) {
  if (!havePreviousDepth) {
    havePreviousDepth = true;
    previousDepthCm = depthCm;
    identicalDepthCount = 0;
    return false;
  }
  if (depthCm == previousDepthCm) {
    if (identicalDepthCount < UINT8_MAX) {
      ++identicalDepthCount;
    }
  } else {
    identicalDepthCount = 0;
  }
  previousDepthCm = depthCm;
  return identicalDepthCount >= 2;
}

bool applyInterlock(bool requestedOpen, const Reading &reading) {
  const float minimumDepth =
      isFlowering(reading.stage) ? kFloweringMinimumDepthCm : kAwdMinimumDepthCm;
  const float openProjection =
      reading.depthCm + kOpenDepthChangeCmPerHour;
  const float closeProjection =
      reading.depthCm + kClosedDepthChangeCmPerHour;

  if (openProjection > kMaximumDepthLimitCm) {
    return false;
  }
  if (closeProjection < minimumDepth) {
    return true;
  }
  return requestedOpen;
}

void processSample(char **tokens, uint8_t count) {
  Reading reading{};
  if (!parseReading(tokens, count, reading)) {
    enterSafeMode("invalid_sample_format");
    return;
  }

  const char *reason = nullptr;
  if (!passesValidityChecks(reading, reason)) {
    enterSafeMode(reason);
    return;
  }
  if (isFrozenDepth(reading.depthCm)) {
    enterSafeMode("frozen_depth");
    return;
  }

  gateOpen = applyInterlock(reading.requestedOpen, reading);
  safeMode = false;
  updateIndicators();
  Serial.print("RESULT gate=");
  Serial.print(gateOpen ? "OPEN" : "CLOSED");
  Serial.println(" safe=0 reason=ok");
}

void resetController() {
  gateOpen = false;
  safeMode = false;
  havePreviousDepth = false;
  previousDepthCm = 0.0F;
  identicalDepthCount = 0;
  updateIndicators();
  Serial.println("RESET gate=CLOSED safe=0");
}

void printHelp() {
  Serial.println("Commands:");
  Serial.println("  SAMPLE <stage> <depth_cm> <request_open:0|1> <age_ms> <fault:0|1>");
  Serial.println("  stage: TRANSPLANT | VEG | FLOWER | RIPEN");
  Serial.println("  RESET");
  Serial.println("Example: SAMPLE VEG -15 1 0 0");
  Serial.println("The requested action is a test input; the Random Forest is not running on the ESP32.");
}

void processLine(char *line) {
  char *tokens[kMaximumTokens] = {};
  uint8_t count = 0;
  char *savePointer = nullptr;
  for (char *token = strtok_r(line, " \t", &savePointer);
       token != nullptr;
       token = strtok_r(nullptr, " \t", &savePointer)) {
    if (count >= kMaximumTokens) {
      enterSafeMode("too_many_fields");
      return;
    }
    tokens[count++] = token;
  }
  if (count == 0) {
    return;
  }
  if (strcmp(tokens[0], "HELP") == 0 && count == 1) {
    printHelp();
  } else if (strcmp(tokens[0], "RESET") == 0 && count == 1) {
    resetController();
  } else if (strcmp(tokens[0], "SAMPLE") == 0) {
    processSample(tokens, count);
  } else {
    enterSafeMode("unknown_command");
  }
}

void readSerialLines() {
  while (Serial.available() > 0) {
    const char incoming = static_cast<char>(Serial.read());
    if (incoming == '\r') {
      continue;
    }
    if (incoming == '\n') {
      if (inputOverflow) {
        enterSafeMode("command_too_long");
      } else {
        inputLine[inputLength] = '\0';
        processLine(inputLine);
      }
      inputLength = 0;
      inputOverflow = false;
      continue;
    }
    if (inputLength + 1 < sizeof(inputLine)) {
      inputLine[inputLength++] = incoming;
    } else {
      inputOverflow = true;
    }
  }
}

}  // namespace

void setup() {
  pinMode(kGateOpenPin, OUTPUT);
  pinMode(kGateClosedPin, OUTPUT);
  pinMode(kSafeModePin, OUTPUT);
  Serial.begin(115200);
  resetController();
  printHelp();
}

void loop() {
  readSerialLines();
}
