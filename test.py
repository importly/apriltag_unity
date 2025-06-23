import serial
import time
import struct
import random
import threading

## testing script if arudino communication is not working.

# Configure serial port parameters
def open_serial(port: str, baud: int = 115200, timeout: float = 0.1) -> serial.Serial:
    ser = serial.Serial(port, baud, timeout=timeout)
    # wait a moment for Arduino reset
    time.sleep(2.0)
    # flush any startup messages
    ser.reset_input_buffer()
    return ser

# Packet encoding as per Arduino endpoint
def encode_drive_packet(channel: int, speed: float) -> bytes:
    raw = int(((speed + 1.0) / 2.0) * 255)
    raw = max(0, min(255, raw))
    lsb = raw & 0x7F
    msb = (raw >> 7) & 0x7F
    return bytes([0x84, channel, lsb, msb])

# Read and parse Arduino acknowledgement
def read_response(ser: serial.Serial) -> str:
    line = ser.readline().decode('ascii', errors='ignore').strip()
    return line

# Basic test for a single channel with given speeds
def test_speed_sequence(ser: serial.Serial, channel: int, speeds: list, pause: float = 0.2):
    print(f"\nTesting channel {channel} speeds: {speeds}")
    for spd in speeds:
        pkt = encode_drive_packet(channel, spd)
        ser.write(pkt)
        time.sleep(pause)
        resp = read_response(ser)
        print(f"Sent spd={spd:.2f}, Response: {resp}")

# Stress test: random commands at high rate
def stress_test(ser: serial.Serial, channel: int, count: int = 1000, rate_hz: float = 50.0):
    interval = 1.0 / rate_hz
    print(f"\nStarting stress test on channel {channel}: {count} commands @ {rate_hz} Hz")
    for i in range(count):
        spd = random.uniform(-1.0, 1.0)
        pkt = encode_drive_packet(channel, spd)
        ser.write(pkt)
        time.sleep(interval)
    print("Stress test complete.")

# Simultaneous test: ramp left and right in parallel
def parallel_ramp_test(ser: serial.Serial, steps: int = 50, duration: float = 5.0):
    print("\nStarting parallel ramp test")
    def send_ramp(ch):
        for i in range(steps + 1):
            spd = -1.0 + (2.0 * i / steps)
            ser.write(encode_drive_packet(ch, spd))
            time.sleep(duration / steps)
    t_left = threading.Thread(target=send_ramp, args=(0,))
    t_right = threading.Thread(target=send_ramp, args=(1,))
    t_left.start(); t_right.start()
    t_left.join(); t_right.join()
    print("Parallel ramp test complete.")

if __name__ == '__main__':
    # Adjust port for your system
    SERIAL_PORT = 'COM7'  # or '/dev/ttyUSB0' for Linux
    BAUD_RATE = 115200
    ser = open_serial(SERIAL_PORT, BAUD_RATE)

    try:
        # 1. Discrete values test
        test_speed_sequence(ser, 0, [-1.0, -0.5, 0.0, 0.5, 1.0])
        test_speed_sequence(ser, 1, [-1.0, -0.5, 0.0, 0.5, 1.0])

        # 2. Stress test
        stress_test(ser, 0, count=500, rate_hz=100)
        stress_test(ser, 1, count=500, rate_hz=100)

        # 3. Parallel ramps
        parallel_ramp_test(ser)

    finally:
        ser.close()
        print("Serial connection closed.")
