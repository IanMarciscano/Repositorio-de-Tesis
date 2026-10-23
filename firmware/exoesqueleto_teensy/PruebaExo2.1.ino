// ============================================================
//  Exoskeleton Control  –  v2.1
//    Non-blocking PD control + CSV data logging
// ============================================================
//
//  v2.1 changes:
//    - Shoulder ROM limited to -40° / +30°.
//    - Serial output replaced with CSV format:
//      timestamp, both load cells (raw + derivative),
//      both joint angles, AK70 feedback (speed, current, temp).
//    - CSV header printed once at startup.  Capture with any
//      serial terminal (CoolTerm, PuTTY, minicom) → .csv file
//      ready for Excel / MATLAB / Python without post-processing.
//
//  v2.0 changes:
//    - Replaced all blocking while-loops and delay() calls with
//      a single non-blocking loop running at LOOP_INTERVAL_US.
//    - Proportional-Derivative (PD) force→velocity control for
//      both joints replaces bang-bang fixed-step logic.
//    - Exponential Moving Average (EMA) on load cell readings
//      to suppress ADC noise before differentiation.
//    - Independent timing / ready flags per load cell.
//    - Serial.print removed from CAN ISR callback.
//    - Servo driven with writeMicroseconds() for sub-degree
//      resolution and smoother motion.
// ============================================================

// -- Libraries -----------------------------------------------
#include <HX711_ADC.h>
#include <EEPROM.h>
#include <Servo.h>
#include <FlexCAN_T4.h>

// -- Timing --------------------------------------------------
//  Target loop period in microseconds.  5 ms = 200 Hz.
const unsigned long LOOP_INTERVAL_US = 5000;
unsigned long lastLoopUs = 0;

//  Separate timestamps for each load cell read-gate (ms).
unsigned long teE = 0;
unsigned long teS = 0;
const unsigned long LC_READ_INTERVAL = 5;  // ms between LC reads

// -- Load Cells ----------------------------------------------
//  Elbow
const int HX711E_dout = 6;
const int HX711E_sck  = 7;
HX711_ADC LoadCell_E(HX711E_dout, HX711E_sck);
const int calValE_eepromAdress = 0;
const unsigned long EStab_time = 2000;
float ECalVal;
float LoadE     = 0;
float LoadE_raw = 0;
float dLoadE    = 0;

//  Shoulder
const int HX711S_dout = 15;
const int HX711S_sck  = 14;
HX711_ADC LoadCell_S(HX711S_dout, HX711S_sck);
const int calValS_eepromAdress = 4;
const unsigned long SStab_time = 2000;
float SCalVal;
float LoadS     = 0;
float LoadS_raw = 0;
float dLoadS    = 0;

//  EMA filter coefficient  (0 < alpha <= 1).
//  Lower = smoother but more lag.  0.15-0.30 is a good range.
const float EMA_ALPHA = 0.20f;

// -- Servo (Elbow) -------------------------------------------
const int Servo_Signal = 9;
Servo elbow;
float elbowAngle   = 0.0f;            // continuous float position (deg)
const float ELBOW_MIN = 0.0f;
const float ELBOW_MAX = 50.0f;

// Microsecond mapping for a typical 0-180 deg servo
const int SERVO_US_MIN = 544;
const int SERVO_US_MAX = 2400;

// -- AK70-10 (Shoulder) -------------------------------------
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> shoulder;

//  Volatile: written inside CAN ISR callback.
volatile float shoulder_pos  = 0;
volatile float shoulder_spd  = 0;
volatile float shoulder_cur  = 0;
volatile int   shoulder_temp = 0;
volatile int   shoulder_err  = 0;

float shoulder_sp = 0;                 // position setpoint (deg)
const float SHOULDER_MIN = -40.0f;    // ROM limit: max extension (deg)
const float SHOULDER_MAX =  30.0f;    // ROM limit: max flexion  (deg)

// -- PD Control Tuning ----------------------------------------
//
//  velocity_cmd = Kp * forceError  +  Kd * dForce
//
//  forceError is zero inside +-deadzone, linear outside.
//  Units:  Kp -> (deg/s)/gram     Kd -> (deg/s)/(g/ms)
//
//  Tuning procedure:
//    1. Set Kd=0.  Raise Kp until the joint tracks user intent.
//    2. Add Kd to dampen overshoot / oscillation.
//    3. If still oscillating, lower Kp or raise deadzone.
//
//  -- Elbow --
const float ELBOW_DEADZONE = 150.0f;  // grams
const float Kp_elbow       =  4.0f;
const float Kd_elbow       =  2.0f;
const float ELBOW_VEL_MAX  = 30.0f;   // deg/s cap

//  -- Shoulder --
const float SHOULDER_DEADZONE = 400.0f;
const float Kp_shoulder       =  2.0f;
const float Kd_shoulder       =  1.0f;
const float SHOULDER_VEL_MAX  = 15.0f; // deg/s cap

// -- CSV Data Logging (throttled) ----------------------------
//  Rate: 20 Hz default.  Raise to 50 (20 ms) for finer
//  resolution during tuning; lower to 10 (100 ms) for
//  long-duration captures.
const unsigned long CSV_INTERVAL_MS = 50;     // ms between rows
unsigned long lastCsvMs = 0;
bool csvHeaderSent = false;

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  Serial.println("Initializing...");

  ShoulderStartup();
  ServoInitialize();
  LC_ShoulderInitialize();
  LC_ElbowInitialize();

  lastLoopUs = micros();
  Serial.println("Exo ready.");

  // Print CSV header row
  Serial.println();
  Serial.println("t_ms,LoadE_g,dLoadE,ElbowAngle_deg,"
                 "LoadS_g,dLoadS,ShoulderSP_deg,ShoulderPos_deg,"
                 "ShoulderSpd_dps,ShoulderCur_A,ShoulderTemp_C,ShoulderErr");
  csvHeaderSent = true;
}

// ============================================================
//  MAIN LOOP  –  non-blocking, fixed-rate
// ============================================================
void loop() {
  // -- Fixed-rate gating ------------------------------------
  unsigned long nowUs = micros();
  if (nowUs - lastLoopUs < LOOP_INTERVAL_US) return;
  float dt_s = (nowUs - lastLoopUs) * 1e-6f;   // actual dt (seconds)
  lastLoopUs = nowUs;

  // -- 1. Read sensors (non-blocking) -----------------------
  UpdateLoadCellE();
  UpdateLoadCellS();
  shoulder.events();

  // -- 2. Elbow PD control ----------------------------------
  float elbowVel = PDController(LoadE, dLoadE,
                                ELBOW_DEADZONE,
                                Kp_elbow, Kd_elbow,
                                ELBOW_VEL_MAX);

  //  Integrate velocity -> angle and clamp.
  elbowAngle += elbowVel * dt_s;
  elbowAngle  = constrain(elbowAngle, ELBOW_MIN, ELBOW_MAX);

  //  Write with microsecond precision for smooth motion.
  int us = mapFloat(elbowAngle, 0.0f, 180.0f, SERVO_US_MIN, SERVO_US_MAX);
  elbow.writeMicroseconds(us);

  // -- 3. Shoulder PD control --------------------------------
  float shoulderVel = PDController(LoadS, dLoadS,
                                   SHOULDER_DEADZONE,
                                   Kp_shoulder, Kd_shoulder,
                                   SHOULDER_VEL_MAX);

  //  Integrate velocity -> setpoint.
  shoulder_sp += shoulderVel * dt_s;
  shoulder_sp  = constrain(shoulder_sp, SHOULDER_MIN, SHOULDER_MAX);

  //  Command AK70 with moderate speed limit + high accel
  //  so the internal position controller can track smoothly.
  setPosSpeed(shoulder_sp, 300, 60000);

  // -- 4. CSV data logging (non-blocking, throttled) ---------
  unsigned long nowMs = millis();
  if (nowMs - lastCsvMs >= CSV_INTERVAL_MS) {
    lastCsvMs = nowMs;
    LogCSVRow(nowMs);
  }
}

// ============================================================
//  CSV LOGGING
// ============================================================
//  Columns (matching the header printed in setup):
//    t_ms            – millis() timestamp
//    LoadE_g         – elbow load cell (grams, filtered)
//    dLoadE          – elbow force derivative (g/ms)
//    ElbowAngle_deg  – servo command angle
//    LoadS_g         – shoulder load cell (grams, filtered)
//    dLoadS          – shoulder force derivative (g/ms)
//    ShoulderSP_deg  – shoulder setpoint sent to AK70
//    ShoulderPos_deg – shoulder encoder feedback
//    ShoulderSpd_dps – shoulder speed feedback (deg/s)
//    ShoulderCur_A   – shoulder motor current (A)
//    ShoulderTemp_C  – shoulder driver temperature
//    ShoulderErr     – shoulder error code (0 = OK)
//
void LogCSVRow(unsigned long t) {
  // Use a single char buffer + snprintf to minimize the
  // number of Serial.print calls (each one flushes the USB
  // buffer and costs ~50-100 us).
  char buf[160];
  snprintf(buf, sizeof(buf),
           "%lu,%.1f,%.3f,%.1f,%.1f,%.3f,%.1f,%.1f,%.1f,%.2f,%d,%d",
           t,
           (double)LoadE,  (double)dLoadE,  (double)elbowAngle,
           (double)LoadS,  (double)dLoadS,
           (double)shoulder_sp, (double)shoulder_pos,
           (double)shoulder_spd, (double)shoulder_cur,
           (int)shoulder_temp, (int)shoulder_err);
  Serial.println(buf);
}

// ============================================================
//  PD CONTROLLER  (shared by both joints)
// ============================================================
//  Returns a velocity command (deg/s).
//  Positive force -> positive velocity.  Flip Kp sign if your
//  load cell convention is reversed for a given joint.
//
float PDController(float force, float dForce,
                   float deadzone, float Kp, float Kd,
                   float velMax) {
  float error = 0;
  if (force >  deadzone) error = force - deadzone;
  if (force < -deadzone) error = force + deadzone;

  float vel = Kp * error + Kd * dForce;
  return constrain(vel, -velMax, velMax);
}

// ============================================================
//  FLOAT MAP  (Arduino map() is integer-only)
// ============================================================
int mapFloat(float x, float in_min, float in_max,
             int out_min, int out_max) {
  return (int)((x - in_min) * (float)(out_max - out_min) /
               (in_max - in_min) + out_min);
}
