# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Read configurations from all robot joints and save to JSON file.

Usage:
    uv run scripts/motor/read_all_configurations.py
    uv run scripts/motor/read_all_configurations.py -o my_config.json
"""

import argparse
import json
import time

import berkeley_humanoid_lite_lowlevel.recoil as recoil


# =============================================================================
# CAN BUS CONFIGURATION - Update these if CAN IDs change after restart
# =============================================================================
LEFT_LEG_CAN = "can2"
RIGHT_LEG_CAN = "can0"
# =============================================================================


# Joint definitions matching humanoid.py
JOINTS = [
    (LEFT_LEG_CAN,  1,  "left_hip_roll"),
    (LEFT_LEG_CAN,  3,  "left_hip_yaw"),
    (LEFT_LEG_CAN,  5,  "left_hip_pitch"),
    (LEFT_LEG_CAN,  7,  "left_knee_pitch"),
    (LEFT_LEG_CAN,  11, "left_ankle_pitch"),
    (LEFT_LEG_CAN,  13, "left_ankle_roll"),
    (RIGHT_LEG_CAN, 2,  "right_hip_roll"),
    (RIGHT_LEG_CAN, 4,  "right_hip_yaw"),
    (RIGHT_LEG_CAN, 6,  "right_hip_pitch"),
    (RIGHT_LEG_CAN, 8,  "right_knee_pitch"),
    (RIGHT_LEG_CAN, 12, "right_ankle_pitch"),
    (RIGHT_LEG_CAN, 14, "right_ankle_roll"),
]


def read_joint_config(bus, device_id, joint_name):
    """Read configuration from a single joint."""
    print(f"  Reading {joint_name} (ID {device_id})...", end=" ")

    # Check if motor is online
    if not bus.ping(device_id):
        print("OFFLINE")
        return None

    config = {
        "joint_name": joint_name,
        "device_id": device_id,
        "position_controller": {},
        "current_controller": {},
        "powerstage": {},
        "motor": {},
        "encoder": {},
    }

    try:
        config["firmware_version"] = hex(bus._read_parameter_u32(device_id, recoil.Parameter.FIRMWARE_VERSION))
        config["watchdog_timeout"] = bus._read_parameter_u32(device_id, recoil.Parameter.WATCHDOG_TIMEOUT)
        config["fast_frame_frequency"] = bus.read_fast_frame_frequency(device_id)

        config["position_controller"]["gear_ratio"] = bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_GEAR_RATIO)
        config["position_controller"]["position_kp"] = bus.read_position_kp(device_id)
        config["position_controller"]["position_kd"] = bus.read_position_kd(device_id)
        config["position_controller"]["position_ki"] = bus.read_position_ki(device_id)
        config["position_controller"]["velocity_kp"] = bus.read_velocity_kp(device_id)
        config["position_controller"]["velocity_ki"] = bus.read_velocity_ki(device_id)
        config["position_controller"]["torque_limit"] = bus.read_torque_limit(device_id)
        config["position_controller"]["velocity_limit"] = bus.read_velocity_limit(device_id)
        config["position_controller"]["position_limit_upper"] = bus.read_position_limit_upper(device_id)
        config["position_controller"]["position_limit_lower"] = bus.read_position_limit_lower(device_id)
        config["position_controller"]["position_offset"] = bus.read_position_offset(device_id)
        config["position_controller"]["torque_filter_alpha"] = bus.read_torque_filter_alpha(device_id)

        config["current_controller"]["i_limit"] = bus.read_current_limit(device_id)
        config["current_controller"]["i_kp"] = bus.read_current_kp(device_id)
        config["current_controller"]["i_ki"] = bus.read_current_ki(device_id)

        config["powerstage"]["undervoltage_threshold"] = bus._read_parameter_f32(device_id, recoil.Parameter.POWERSTAGE_UNDERVOLTAGE_THRESHOLD)
        config["powerstage"]["overvoltage_threshold"] = bus._read_parameter_f32(device_id, recoil.Parameter.POWERSTAGE_OVERVOLTAGE_THRESHOLD)
        config["powerstage"]["bus_voltage_filter_alpha"] = bus.read_bus_voltage_filter_alpha(device_id)

        config["motor"]["pole_pairs"] = bus.read_motor_pole_pairs(device_id)
        config["motor"]["torque_constant"] = bus.read_motor_torque_constant(device_id)
        config["motor"]["phase_order"] = bus.read_motor_phase_order(device_id)
        config["motor"]["max_calibration_current"] = bus.read_motor_calibration_current(device_id)

        config["encoder"]["cpr"] = bus.read_encoder_cpr(device_id)
        config["encoder"]["position_offset"] = bus.read_encoder_position_offset(device_id)
        config["encoder"]["velocity_filter_alpha"] = bus.read_encoder_velocity_filter_alpha(device_id)
        config["encoder"]["flux_offset"] = bus.read_encoder_flux_offset(device_id)

        print("OK")
        return config

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Read configurations from all robot joints")
    parser.add_argument("-o", "--output", type=str, default="all_joint_configurations.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    print(f"Initializing CAN buses: LEFT_LEG={LEFT_LEG_CAN}, RIGHT_LEG={RIGHT_LEG_CAN}")
    buses = {
        LEFT_LEG_CAN: recoil.Bus(channel=LEFT_LEG_CAN, bitrate=1000000),
        RIGHT_LEG_CAN: recoil.Bus(channel=RIGHT_LEG_CAN, bitrate=1000000),
    }

    all_configs = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "joints": {}
    }

    print("\nReading joint configurations:")
    for channel, device_id, joint_name in JOINTS:
        bus = buses[channel]
        config = read_joint_config(bus, device_id, joint_name)
        if config:
            all_configs["joints"][joint_name] = config
        time.sleep(0.05)  # Small delay between reads

    # Stop buses
    for bus in buses.values():
        bus.stop()

    # Save to file
    with open(args.output, "w") as f:
        json.dump(all_configs, f, indent=2)

    print(f"\nSaved configurations to {args.output}")
    print(f"Successfully read {len(all_configs['joints'])}/{len(JOINTS)} joints")

    # Print summary
    if all_configs["joints"]:
        print("\n--- Summary ---")
        first_joint = list(all_configs["joints"].values())[0]
        print(f"Firmware version: {first_joint.get('firmware_version', 'N/A')}")
        print(f"Gear ratio: {first_joint['position_controller'].get('gear_ratio', 'N/A')}")
        print(f"Position Kp: {first_joint['position_controller'].get('position_kp', 'N/A')}")
        print(f"Position Kd: {first_joint['position_controller'].get('position_kd', 'N/A')}")
        print(f"Torque limit: {first_joint['position_controller'].get('torque_limit', 'N/A')}")


if __name__ == "__main__":
    main()
