# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Actuator Monitoring Script

Displays real-time status of all actuators including:
- Error flags (over-temperature, over-voltage, under-voltage, etc.)
- Bus voltage
- Current draw (I_Q)
- IMU temperature

Usage:
    uv run scripts/motor/monitor_actuators.py
"""

import sys
import time

import berkeley_humanoid_lite_lowlevel.recoil as recoil
from berkeley_humanoid_lite_lowlevel.robot.imu import SerialImu, Baudrate


# Error flag definitions from motor_controller_conf.h
ERROR_FLAGS = {
    0b0000000000000001: "POWERSTAGE_ERR",
    0b0000000000000010: "ENCODER_NOT_READY",
    0b0000000000000100: "ENCODER_ERR",
    0b0000000000001000: "OVER_CURRENT",
    0b0000000000010000: "OVER_VOLTAGE",
    0b0000000000100000: "UNDER_VOLTAGE",
    0b0000000001000000: "SAFETY_WATCHDOG",
    0b0000000010000000: "CALIBRATION_ERR",
    0b0000000100000000: "INVALID_MODE",
    0b0000001000000000: "OVER_TEMPERATURE",
}

# Joint configuration (matches humanoid.py)
JOINTS = [
    ("can1", 1, "left_hip_roll"),
    ("can1", 3, "left_hip_yaw"),
    ("can1", 5, "left_hip_pitch"),
    ("can1", 7, "left_knee_pitch"),
    ("can1", 11, "left_ankle_pitch"),
    ("can1", 13, "left_ankle_roll"),
    ("can0", 2, "right_hip_roll"),
    ("can0", 4, "right_hip_yaw"),
    ("can0", 6, "right_hip_pitch"),
    ("can0", 8, "right_knee_pitch"),
    ("can0", 12, "right_ankle_pitch"),
    ("can0", 14, "right_ankle_roll"),
]


def decode_errors(error_value):
    """Decode error flags into human-readable strings."""
    if error_value == 0:
        return "OK"

    errors = []
    for flag, name in ERROR_FLAGS.items():
        if error_value & flag:
            errors.append(name)

    return ", ".join(errors) if errors else f"UNKNOWN(0x{error_value:04X})"


def main():
    print("\n=== Actuator Monitor ===\n")

    # Initialize CAN buses
    print("Initializing CAN buses...")
    buses = {}
    try:
        buses["can1"] = recoil.Bus(channel="can1", bitrate=1000000)
        buses["can2"] = recoil.Bus(channel="can2", bitrate=1000000)
    except Exception as e:
        print(f"Error initializing CAN buses: {e}")
        return

    # Initialize IMU
    print("Initializing IMU...")
    imu = None
    try:
        imu = SerialImu(port="/dev/ttyACM0", baudrate=Baudrate.BAUD_1000000)
        imu.run_forever()
        time.sleep(0.5)  # Wait for IMU to start
    except Exception as e:
        print(f"Warning: Could not initialize IMU: {e}")

    print("\nMonitoring actuators (Ctrl+C to stop)...\n")

    try:
        while True:
            # Clear screen and move cursor to top
            print("\033[H\033[J", end="")  # ANSI escape: clear screen

            print("=" * 100)
            print(f"{'ACTUATOR MONITOR':^100}")
            print(f"{'Time: ' + time.strftime('%H:%M:%S'):^100}")
            print("=" * 100)

            # IMU temperature
            if imu is not None:
                print(f"\nIMU Temperature: {imu.temperature:.1f} C")
            else:
                print("\nIMU: Not available")

            # Table header
            print("\n" + "-" * 100)
            print(f"{'Joint':<25} {'ID':>4} {'Bus V':>8} {'I_Q':>8} {'Torque':>8} {'Status':<40}")
            print("-" * 100)

            for can_channel, device_id, joint_name in JOINTS:
                bus = buses[can_channel]

                try:
                    # Read error status
                    error = bus.read_error(device_id)
                    error_str = decode_errors(error) if error is not None else "NO_RESP"

                    # Read bus voltage
                    bus_voltage = bus.read_bus_voltage(device_id)
                    voltage_str = f"{bus_voltage:.1f}V" if bus_voltage is not None else "N/A"

                    # Read I_Q (current in Q axis = torque-producing current)
                    i_q = bus.read_i_q_measured(device_id)
                    i_q_str = f"{i_q:.2f}A" if i_q is not None else "N/A"

                    # Read torque measured
                    torque = bus.read_torque_measured(device_id)
                    torque_str = f"{torque:.2f}Nm" if torque is not None else "N/A"

                    # Color coding for errors
                    if error is not None and error != 0:
                        status_color = "\033[91m"  # Red
                    else:
                        status_color = "\033[92m"  # Green
                    reset_color = "\033[0m"

                    print(f"{joint_name:<25} {device_id:>4} {voltage_str:>8} {i_q_str:>8} {torque_str:>8} {status_color}{error_str:<40}{reset_color}")

                except Exception as e:
                    print(f"{joint_name:<25} {device_id:>4} {'ERROR':>8} {'---':>8} {'---':>8} {str(e):<40}")

                time.sleep(0.005)  # Small delay between reads

            print("-" * 100)
            print("\nPress Ctrl+C to stop")

            time.sleep(0.5)  # Update every 500ms

    except KeyboardInterrupt:
        print("\n\nStopping monitor...")

    finally:
        # Cleanup
        for bus in buses.values():
            bus.stop()

        if imu is not None:
            imu.stop()

        print("Done.")


if __name__ == "__main__":
    main()
