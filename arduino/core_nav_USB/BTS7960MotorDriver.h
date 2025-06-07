#ifndef BTS7960MotorDriverRC_h
#define BTS7960MotorDriverRC_h

#include <Arduino.h>
#include "driver/ledc.h" // ESP-IDF style LEDC API

/**
 * @enum Direction
 * @brief Represents the motor direction (Stop, Forward, or Reverse).
 */
enum Direction {
  DIR_STOP = 0,
  DIR_FORWARD,
  DIR_REVERSE
};

/**
 * @class BTS7960MotorDriverRC
 * @brief Manages two-channel BTS7960 motor driver with optional RC override.
 * 
 * Features:
 *  - PWM-based motor control (ESP32 LEDC).
 *  - Forward/Reverse with ramping to avoid abrupt changes.
 *  - RC override on three channels:
 *    * channel0 => left speed
 *    * channel1 => right speed
 *    * channel2 => mode (above threshold => manual override)
 */
class BTS7960MotorDriverRC {
public:
  /**
   * @brief Constructor
   *
   * @param rpwm1, lpwm1, ren1, len1  Pins for motor channel 1
   * @param rpwm2, lpwm2, ren2, len2  Pins for motor channel 2
   * @param rcPin0, rcPin1, rcPin2    RC input pins
   * @param rcBypassThreshold         If rcPin2's pulse > threshold => manual override
   * @param pwmFreq                   LEDC frequency (Hz)
   * @param pwmBits                   LEDC duty resolution (bits)
   * @param accelStep                 Ramping acceleration step
   * @param decelStep                 Ramping deceleration step
   * @param rampInterval              Milliseconds between ramp updates
   * @param directionWait             ms to hold at zero speed before reversing
   */
  BTS7960MotorDriverRC(uint8_t rpwm1, uint8_t lpwm1, uint8_t ren1, uint8_t len1,
                       uint8_t rpwm2, uint8_t lpwm2, uint8_t ren2, uint8_t len2,
                       uint8_t rcPin0, uint8_t rcPin1, uint8_t rcPin2,
                       int rcBypassThreshold = 1700,
                       uint32_t pwmFreq      = 25000,
                       uint8_t  pwmBits      = 8,
                       int      accelStep    = 40,
                       int      decelStep    = 40,
                       uint32_t rampInterval = 1,
                       uint32_t directionWait= 0)
  : _rpwm1(rpwm1), _lpwm1(lpwm1), _ren1(ren1), _len1(len1),
    _rpwm2(rpwm2), _lpwm2(lpwm2), _ren2(ren2), _len2(len2),
    _rcPin0(rcPin0), _rcPin1(rcPin1), _rcPin2(rcPin2),
    _rcBypassThreshold(rcBypassThreshold),
    _pwmFreq(pwmFreq), _pwmBits(pwmBits),
    _accelStep(accelStep), _decelStep(decelStep),
    _rampIntervalMs(rampInterval), _directionWaitMs(directionWait)
  {
    // Channel 1 (motor) initial states
    _currentSpeed1 = 0;
    _currentDir1   = DIR_STOP;
    _targetSpeed1  = 0;
    _targetDir1    = DIR_STOP;
    _lastRamp1     = 0;
    _waitingZero1  = false;

    // Channel 2 (motor) initial states
    _currentSpeed2 = 0;
    _currentDir2   = DIR_STOP;
    _targetSpeed2  = 0;
    _targetDir2    = DIR_STOP;
    _lastRamp2     = 0;
    _waitingZero2  = false;

    // RC initial values
    _rcValue0 = 1500;
    _rcValue1 = 1500;
    _rcValue2 = 1000; 
    _rcStart0 = 0;
    _rcStart1 = 0;
    _rcStart2 = 0;
  }

  /**
   * @brief Initializes motor driver (LEDC) and sets up RC interrupts.
   *        Must be called once in Arduino setup().
   */
  void setup() {
    // Enable driver pins
    pinMode(_ren1, OUTPUT); digitalWrite(_ren1, HIGH);
    pinMode(_len1, OUTPUT); digitalWrite(_len1, HIGH);
    pinMode(_ren2, OUTPUT); digitalWrite(_ren2, HIGH);
    pinMode(_len2, OUTPUT); digitalWrite(_len2, HIGH);

    // Configure LEDC Timer
    ledc_timer_config_t ledc_timer = {
      .speed_mode      = LEDC_HIGH_SPEED_MODE,
      .duty_resolution = (ledc_timer_bit_t)_pwmBits,
      .timer_num       = LEDC_TIMER_0,
      .freq_hz         = _pwmFreq,
      .clk_cfg         = LEDC_AUTO_CLK
    };
    ledc_timer_config(&ledc_timer);

    // Configure 4 channels for the two motors
    initChannel(LEDC_CHANNEL_RPWM1, _rpwm1);
    initChannel(LEDC_CHANNEL_LPWM1, _lpwm1);
    initChannel(LEDC_CHANNEL_RPWM2, _rpwm2);
    initChannel(LEDC_CHANNEL_LPWM2, _lpwm2);

    // Start with zero duty
    setDuty(LEDC_CHANNEL_RPWM1, 0);
    setDuty(LEDC_CHANNEL_LPWM1, 0);
    setDuty(LEDC_CHANNEL_RPWM2, 0);
    setDuty(LEDC_CHANNEL_LPWM2, 0);

    // RC pins
    pinMode(_rcPin0, INPUT);
    pinMode(_rcPin1, INPUT);
    pinMode(_rcPin2, INPUT);

    // Attach interrupts using argument so each ISR can reference 'this'
    attachInterruptArg(_rcPin0, _isrPin0, this, CHANGE);
    attachInterruptArg(_rcPin1, _isrPin1, this, CHANGE);
    attachInterruptArg(_rcPin2, _isrPin2, this, CHANGE);
  }

  /**
   * @brief Must be called frequently in loop().
   *        - Checks RC override
   *        - Ramps motor speeds
   */
  void loop() {
    bool manualOverride = (_rcValue2 > _rcBypassThreshold);

    if (manualOverride) {
      float leftSpeed  = mapRCtoSpeed(_rcValue0);
      float rightSpeed = mapRCtoSpeed(_rcValue1);
      setMotorTargets(leftSpeed, rightSpeed);
    }
    // If not manual, rely on any last setTargetSpeeds() calls

    // Ramp both motors
    rampMotor1();
    rampMotor2();
  }

  /**
   * @brief Set the motor speeds in [-1..+1], unless RC override is active.
   */
  void setTargetSpeeds(float left, float right) {
    if (_rcValue2 <= _rcBypassThreshold) {
      // Only apply if NOT in manual override
      setMotorTargets(left, right);
    }
  }

private:
  /* --------------------------------------------------------------------------
     Motor pin assignments
     --------------------------------------------------------------------------*/
  uint8_t _rpwm1, _lpwm1, _ren1, _len1;
  uint8_t _rpwm2, _lpwm2, _ren2, _len2;

  // RC pins and threshold
  uint8_t _rcPin0, _rcPin1, _rcPin2;
  int     _rcBypassThreshold;

  // LEDC config
  uint32_t _pwmFreq;
  uint8_t  _pwmBits;

  // Ramping config
  int      _accelStep;
  int      _decelStep;
  uint32_t _rampIntervalMs;
  uint32_t _directionWaitMs;

  /* --------------------------------------------------------------------------
     Motor states
     --------------------------------------------------------------------------*/
  // Motor #1
  int       _currentSpeed1;  // 0..255
  Direction _currentDir1;
  int       _targetSpeed1;   // 0..255
  Direction _targetDir1;
  unsigned long _lastRamp1;
  bool      _waitingZero1;

  // Motor #2
  int       _currentSpeed2;  // 0..255
  Direction _currentDir2;
  int       _targetSpeed2;   // 0..255
  Direction _targetDir2;
  unsigned long _lastRamp2;
  bool      _waitingZero2;

  // LEDC channel constants
  static constexpr ledc_channel_t LEDC_CHANNEL_RPWM1 = LEDC_CHANNEL_0;
  static constexpr ledc_channel_t LEDC_CHANNEL_LPWM1 = LEDC_CHANNEL_1;
  static constexpr ledc_channel_t LEDC_CHANNEL_RPWM2 = LEDC_CHANNEL_2;
  static constexpr ledc_channel_t LEDC_CHANNEL_LPWM2 = LEDC_CHANNEL_3;

  /* --------------------------------------------------------------------------
     RC reading data
     --------------------------------------------------------------------------*/
  // Current pulse width (microseconds)
  volatile int _rcValue0;
  volatile int _rcValue1;
  volatile int _rcValue2;

  // For measuring start times
  volatile unsigned long _rcStart0;
  volatile unsigned long _rcStart1;
  volatile unsigned long _rcStart2;

  /* --------------------------------------------------------------------------
     Private helper methods
     --------------------------------------------------------------------------*/

  /**
   * @brief Initialize one LEDC channel with a given pin.
   */
  void initChannel(ledc_channel_t chan, int pin) {
    ledc_channel_config_t cfg = {
      .gpio_num       = pin,
      .speed_mode     = LEDC_HIGH_SPEED_MODE,
      .channel        = chan,
      .intr_type      = LEDC_INTR_DISABLE,
      .timer_sel      = LEDC_TIMER_0,
      .duty           = 0,
      .hpoint         = 0
    };
    ledc_channel_config(&cfg);
  }

  /**
   * @brief Safely set LEDC duty cycle for an 8-bit resolution.
   */
  void setDuty(ledc_channel_t chan, int duty) {
    if (duty < 0)   duty = 0;
    if (duty > 255) duty = 255;
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, chan, duty);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, chan);
  }

  /**
   * @brief Static interrupt stub for RC channel 0.
   */
  static void _isrPin0(void* arg) {
    BTS7960MotorDriverRC* self = static_cast<BTS7960MotorDriverRC*>(arg);
    self->handleRCInterrupt0();
  }

  /**
   * @brief Static interrupt stub for RC channel 1.
   */
  static void _isrPin1(void* arg) {
    BTS7960MotorDriverRC* self = static_cast<BTS7960MotorDriverRC*>(arg);
    self->handleRCInterrupt1();
  }

  /**
   * @brief Static interrupt stub for RC channel 2.
   */
  static void _isrPin2(void* arg) {
    BTS7960MotorDriverRC* self = static_cast<BTS7960MotorDriverRC*>(arg);
    self->handleRCInterrupt2();
  }

  /**
   * @brief Actual handler for RC channel 0. Measures the pulse width.
   */
  void handleRCInterrupt0() {
    bool level = digitalRead(_rcPin0);
    if (level) {
      _rcStart0 = micros();
    } else {
      _rcValue0 = (int)(micros() - _rcStart0);
    }
  }

  /**
   * @brief Actual handler for RC channel 1. Measures the pulse width.
   */
  void handleRCInterrupt1() {
    bool level = digitalRead(_rcPin1);
    if (level) {
      _rcStart1 = micros();
    } else {
      _rcValue1 = (int)(micros() - _rcStart1);
    }
  }

  /**
   * @brief Actual handler for RC channel 2. Measures the pulse width.
   */
  void handleRCInterrupt2() {
    bool level = digitalRead(_rcPin2);
    if (level) {
      _rcStart2 = micros();
    } else {
      _rcValue2 = (int)(micros() - _rcStart2);
    }
  }

  /**
   * @brief Convert RC pulse width in [1000..2000] to speed in [-1..+1].
   */
  float mapRCtoSpeed(int pulseUs) {
    // Typical: 1000us => -1, 1500us => 0, 2000us => +1
    // Adjust as needed for your transmitter
    const float inMin = 1000.0f;
    const float inMid = 1500.0f;
    const float inMax = 2000.0f;

    if (pulseUs <= inMin)   return -1.0f; 
    if (pulseUs >= inMax)   return +1.0f;

    if (pulseUs < inMid) {
      // [1000..1500] => [-1..0]
      float ratio = (pulseUs - 1000.0f) / 500.0f;  // 0..1
      return -1.0f + ratio;                       // -1..0
    } else {
      // [1500..2000] => [0..+1]
      float ratio = (pulseUs - 1500.0f) / 500.0f;  // 0..1
      return ratio;                                // 0..+1
    }
  }

  /**
   * @brief Update internal target speed/direction from a [-1..+1] speed command.
   */
  void setMotorTargets(float left, float right) {
    left  = constrain(left,  -1.0f, 1.0f);
    right = constrain(right, -1.0f, 1.0f);

    // Convert magnitude to 0..255
    int speedL = (int)(fabs(left)  * 255.0f);
    int speedR = (int)(fabs(right) * 255.0f);

    Direction dirL = (left > 0) ? DIR_FORWARD
                    : (left < 0) ? DIR_REVERSE
                                 : DIR_STOP;
    Direction dirR = (right> 0) ? DIR_FORWARD
                    : (right< 0) ? DIR_REVERSE
                                 : DIR_STOP;

    _targetSpeed1 = speedL;
    _targetDir1   = dirL;

    _targetSpeed2 = speedR;
    _targetDir2   = dirR;
  }

  /**
   * @brief Ramp logic for motor 1.
   */
  void rampMotor1() {
    unsigned long now = millis();
    if ((now - _lastRamp1) < _rampIntervalMs) {
      return;
    }
    _lastRamp1 = now;

    // If direction changed => ramp down to 0
    if (_currentDir1 != _targetDir1) {
      if (_currentSpeed1 > 0) {
        int step = getDecelStep(_currentSpeed1);
        _currentSpeed1 -= step;
        if (_currentSpeed1 < 0) _currentSpeed1 = 0;
        applyMotorOutput1(_currentDir1, _currentSpeed1);

        if (_currentSpeed1 == 0 && _directionWaitMs > 0) {
          _waitingZero1 = true;
        }
        return;
      } else {
        if (_waitingZero1) {
          static unsigned long zeroStart1 = 0;
          if (zeroStart1 == 0) zeroStart1 = now;

          if ((now - zeroStart1) < _directionWaitMs) {
            return;
          } else {
            _waitingZero1 = false;
            zeroStart1 = 0;
          }
        }
        // Switch direction
        _currentDir1 = _targetDir1;
      }
    }

    // Ramp speed
    if (_currentSpeed1 < _targetSpeed1) {
      int step = getAccelStep(_targetSpeed1 - _currentSpeed1);
      _currentSpeed1 += step;
      if (_currentSpeed1 > _targetSpeed1) {
        _currentSpeed1 = _targetSpeed1;
      }
    }
    else if (_currentSpeed1 > _targetSpeed1) {
      int step = getDecelStep(_currentSpeed1 - _targetSpeed1);
      _currentSpeed1 -= step;
      if (_currentSpeed1 < _targetSpeed1) {
        _currentSpeed1 = _targetSpeed1;
      }
    }
    applyMotorOutput1(_currentDir1, _currentSpeed1);
  }

  /**
   * @brief Actually set the PWM outputs for motor 1.
   */
  void applyMotorOutput1(Direction dir, int speedVal) {
    switch (dir) {
      case DIR_FORWARD:
        setDuty(LEDC_CHANNEL_RPWM1, speedVal);
        setDuty(LEDC_CHANNEL_LPWM1, 0);
        break;
      case DIR_REVERSE:
        setDuty(LEDC_CHANNEL_RPWM1, 0);
        setDuty(LEDC_CHANNEL_LPWM1, speedVal);
        break;
      default: // STOP
        setDuty(LEDC_CHANNEL_RPWM1, 0);
        setDuty(LEDC_CHANNEL_LPWM1, 0);
        break;
    }
  }

  /**
   * @brief Ramp logic for motor 2.
   */
  void rampMotor2() {
    unsigned long now = millis();
    if ((now - _lastRamp2) < _rampIntervalMs) {
      return;
    }
    _lastRamp2 = now;

    // If direction changed => ramp to 0 first
    if (_currentDir2 != _targetDir2) {
      if (_currentSpeed2 > 0) {
        int step = getDecelStep(_currentSpeed2);
        _currentSpeed2 -= step;
        if (_currentSpeed2 < 0) _currentSpeed2 = 0;
        applyMotorOutput2(_currentDir2, _currentSpeed2);

        if (_currentSpeed2 == 0 && _directionWaitMs > 0) {
          _waitingZero2 = true;
        }
        return;
      } else {
        if (_waitingZero2) {
          static unsigned long zeroStart2 = 0;
          if (zeroStart2 == 0) zeroStart2 = now;

          if ((now - zeroStart2) < _directionWaitMs) {
            return;
          } else {
            _waitingZero2 = false;
            zeroStart2 = 0;
          }
        }
        _currentDir2 = _targetDir2;
      }
    }

    // Ramp speed
    if (_currentSpeed2 < _targetSpeed2) {
      int step = getAccelStep(_targetSpeed2 - _currentSpeed2);
      _currentSpeed2 += step;
      if (_currentSpeed2 > _targetSpeed2) {
        _currentSpeed2 = _targetSpeed2;
      }
    }
    else if (_currentSpeed2 > _targetSpeed2) {
      int step = getDecelStep(_currentSpeed2 - _targetSpeed2);
      _currentSpeed2 -= step;
      if (_currentSpeed2 < _targetSpeed2) {
        _currentSpeed2 = _targetSpeed2;
      }
    }
    applyMotorOutput2(_currentDir2, _currentSpeed2);
  }

  /**
   * @brief Actually set the PWM outputs for motor 2.
   */
  void applyMotorOutput2(Direction dir, int speedVal) {
    switch (dir) {
      case DIR_FORWARD:
        setDuty(LEDC_CHANNEL_RPWM2, speedVal);
        setDuty(LEDC_CHANNEL_LPWM2, 0);
        break;
      case DIR_REVERSE:
        setDuty(LEDC_CHANNEL_RPWM2, 0);
        setDuty(LEDC_CHANNEL_LPWM2, speedVal);
        break;
      default: // STOP
        setDuty(LEDC_CHANNEL_RPWM2, 0);
        setDuty(LEDC_CHANNEL_LPWM2, 0);
        break;
    }
  }

  // Ramping step helpers:
  int getAccelStep(int diff) {
    return _accelStep;
  }
  int getDecelStep(int diff) {
    return _decelStep;
  }
};

#endif
