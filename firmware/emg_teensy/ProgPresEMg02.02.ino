#include <Arduino.h>
#include <IntervalTimer.h>
#include <string.h>

static constexpr uint32_t FS_HZ = 1000;
static constexpr uint32_t SERIAL_BAUD = 2000000;

// ==================== MAPEO DE PINES ====================
// Teensy 4.0 (número de pin físico -> canal analógico):
// Pin 17 = A3  -> RECT bíceps
// Pin 16 = A2  -> RECT tríceps
// Pin 15 = A1  -> RAW  bíceps
// Pin 14 = A0  -> RAW  tríceps
static constexpr uint8_t PIN_CH0 = A3;  // pin 17: RECT bíceps
static constexpr uint8_t PIN_CH1 = A2;  // pin 16: RECT tríceps
static constexpr uint8_t PIN_CH2 = A1;  // pin 15: RAW  bíceps
static constexpr uint8_t PIN_CH3 = A0;  // pin 14: RAW  tríceps
// ================================================================================================

static constexpr uint8_t ADC_BITS = 12;

static volatile bool streaming = false;
static uint32_t t0_us = 0;
static volatile uint32_t seq_counter = 0;

struct EmgSample {
  uint32_t t_us;
  uint32_t seq;
  uint16_t c0, c1, c2, c3;
};

static constexpr uint16_t BUF_SIZE = 2048;
static EmgSample ring[BUF_SIZE];
static volatile uint16_t wIdx = 0;
static volatile uint16_t rIdx = 0;

IntervalTimer sampler;

static constexpr uint8_t MAGIC0 = 0xA5;
static constexpr uint8_t MAGIC1 = 0x5A;
static constexpr uint8_t PKT_EMG = 0x01;

#pragma pack(push, 1)
struct EmgPacket {
  uint8_t m0, m1, type;
  uint32_t t_us;
  uint32_t seq;
  uint16_t c0, c1, c2, c3;
};
#pragma pack(pop)

void ISR_sample() {
  if (!streaming) return;

  uint16_t c0 = analogRead(PIN_CH0);
  uint16_t c1 = analogRead(PIN_CH1);
  uint16_t c2 = analogRead(PIN_CH2);
  uint16_t c3 = analogRead(PIN_CH3);

  uint16_t localW = wIdx;
  uint16_t nextW = (localW + 1) & (BUF_SIZE - 1);
  if (nextW == rIdx) return;

  ring[localW].t_us = (uint32_t)(micros() - t0_us);
  ring[localW].seq  = seq_counter++;
  ring[localW].c0 = c0;
  ring[localW].c1 = c1;
  ring[localW].c2 = c2;
  ring[localW].c3 = c3;

  asm volatile("" ::: "memory");
  wIdx = nextW;
}

static void handle_line(char *line) {
  // Quitar '\r' si llega "START\r\n" desde Windows
  size_t L = strlen(line);
  if (L > 0 && line[L - 1] == '\r') line[L - 1] = '\0';

  if (strcmp(line, "START") == 0) {
    noInterrupts();
    wIdx = 0; rIdx = 0;
    t0_us = micros();
    seq_counter = 0;
    streaming = true;
    interrupts();
    Serial.println("OK START");
    return;
  }
  if (strcmp(line, "STOP") == 0) {
    streaming = false;
    Serial.println("OK STOP");
    return;
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 3000) {}

  analogReadResolution(ADC_BITS);
  analogReadAveraging(1);

  uint32_t period_us = 1000000UL / FS_HZ;
  sampler.begin(ISR_sample, period_us);

  Serial.println("EMG READY");
  Serial.println("Send START or STOP");
}

void loop() {
  static char linebuf[64];
  static uint8_t idx = 0;

  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      linebuf[idx] = '\0';
      idx = 0;
      handle_line(linebuf);
    } else if (idx < sizeof(linebuf) - 1) {
      linebuf[idx++] = c;
    }
  }

  while (rIdx != wIdx) {
    uint16_t localR = rIdx;
    EmgSample s = ring[localR];
    rIdx = (localR + 1) & (BUF_SIZE - 1);

    EmgPacket p;
    p.m0 = MAGIC0; p.m1 = MAGIC1; p.type = PKT_EMG;
    p.t_us = s.t_us;
    p.seq  = s.seq;
    p.c0 = s.c0; p.c1 = s.c1; p.c2 = s.c2; p.c3 = s.c3;

    Serial.write((uint8_t*)&p, sizeof(p));
  }
}
