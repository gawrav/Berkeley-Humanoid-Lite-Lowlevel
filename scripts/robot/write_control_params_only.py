# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.
#
# SAFE Configuration Script - Only writes control parameters
# Does NOT touch:
#   - motor physical params (pole_pairs, torque_constant) - correct from firmware
#   - current_controller gains (i_kp, i_ki) - computed from motor profile
#   - encoder flux_offset - requires electrical calibration
#
# Usage:
#   uv run scripts/robot/write_control_params_only.py

import time

import berkeley_humanoid_lite_lowlevel.recoil as recoil
from berkeley_humanoid_lite_lowlevel.robot import Humanoid

# ============ CONFIGURATION ============
# Control parameters for leg joints (biped)
# These are the parameters that need to be restored after motor reset

LEG_CONTROL_PARAMS = {
    "gear_ratio": -15.0,      # 15:1 reduction, negative for direction
    # "position_kp": 20.0,      # Position P gain
    # "position_ki": 0.0,       # Position I gain
    # "velocity_kp": 2.0,       # Velocity P gain (acts as position D)
    # "velocity_ki": 0.0,       # Velocity I gain
    # "torque_limit": 6.0,      # Nm at output shaft
    # "velocity_limit": 20.0,   # rad/s
    # "torque_filter_alpha": 0.27,  # Low-pass filter
}

# List of leg joint names (biped configuration)
LEG_JOINTS = [
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

# ============ END CONFIGURATION ============

STORE_TO_FLASH = True
DELAY_T = 0.1


def main():
    print("=" * 60)
    print("SAFE Configuration Script - Control Parameters Only")
    print("=" * 60)
    print()
    print("This script will write ONLY gear_ratio:")
    print(f"  gear_ratio:    {LEG_CONTROL_PARAMS['gear_ratio']}")
    # print(f"  position_kp:   {LEG_CONTROL_PARAMS['position_kp']}")
    # print(f"  position_ki:   {LEG_CONTROL_PARAMS['position_ki']}")
    # print(f"  velocity_kp:   {LEG_CONTROL_PARAMS['velocity_kp']}")
    # print(f"  velocity_ki:   {LEG_CONTROL_PARAMS['velocity_ki']}")
    # print(f"  torque_limit:  {LEG_CONTROL_PARAMS['torque_limit']}")
    # print(f"  velocity_limit: {LEG_CONTROL_PARAMS['velocity_limit']}")
    print()
    print("It will NOT touch:")
    print("  - Motor physical params (pole_pairs, torque_constant)")
    print("  - Current controller gains (i_kp, i_ki)")
    print("  - Encoder flux_offset (needs electrical calibration)")
    print()

    input("Press Enter to continue, or Ctrl+C to abort...")
    print()

    robot = Humanoid()

    # First, ping all joints
    print("Checking all joints...")
    for entry in robot.joints:
        bus, joint_id, joint_name = entry
        if joint_name not in LEG_JOINTS:
            continue

        result = bus.ping(joint_id)
        if not result:
            print(f"  ERROR: {joint_name} (id={joint_id}) not responding!")
            robot.stop()
            return
        print(f"  OK: {joint_name} (id={joint_id})")
        time.sleep(0.05)

    print()
    print("All joints responding. Writing control parameters...")
    print()

    # Write parameters to each joint
    for entry in robot.joints:
        bus, joint_id, joint_name = entry
        if joint_name not in LEG_JOINTS:
            continue

        print(f"Configuring {joint_name} (id={joint_id}):")

        # Read current gear_ratio to show change
        current_gr = bus._read_parameter_f32(joint_id, recoil.Parameter.POSITION_CONTROLLER_GEAR_RATIO)
        print(f"  current gear_ratio: {current_gr}")

        # Write control parameters
        val = LEG_CONTROL_PARAMS["gear_ratio"]
        print(f"  setting gear_ratio to {val}")
        bus._write_parameter_f32(joint_id, recoil.Parameter.POSITION_CONTROLLER_GEAR_RATIO, val)
        time.sleep(DELAY_T)

        # Step 1: Verify SDO write succeeded by reading back
        verified_gr = bus._read_parameter_f32(joint_id, recoil.Parameter.POSITION_CONTROLLER_GEAR_RATIO)
        if verified_gr != val:
            print(f"  ERROR: SDO write verification failed! Expected {val}, got {verified_gr}")
            robot.stop()
            return
        print(f"  verified gear_ratio after SDO write: {verified_gr}")

        # val = LEG_CONTROL_PARAMS["position_kp"]
        # print(f"  setting position_kp to {val}")
        # bus.write_position_kp(joint_id, val)
        # time.sleep(DELAY_T)

        # val = LEG_CONTROL_PARAMS["position_ki"]
        # print(f"  setting position_ki to {val}")
        # bus.write_position_ki(joint_id, val)
        # time.sleep(DELAY_T)

        # val = LEG_CONTROL_PARAMS["velocity_kp"]
        # print(f"  setting velocity_kp to {val}")
        # bus.write_velocity_kp(joint_id, val)
        # time.sleep(DELAY_T)

        # val = LEG_CONTROL_PARAMS["velocity_ki"]
        # print(f"  setting velocity_ki to {val}")
        # bus.write_velocity_ki(joint_id, val)
        # time.sleep(DELAY_T)

        # val = LEG_CONTROL_PARAMS["torque_limit"]
        # print(f"  setting torque_limit to {val}")
        # bus.write_torque_limit(joint_id, val)
        # time.sleep(DELAY_T)

        # val = LEG_CONTROL_PARAMS["velocity_limit"]
        # print(f"  setting velocity_limit to {val}")
        # bus.write_velocity_limit(joint_id, val)
        # time.sleep(DELAY_T)

        # val = LEG_CONTROL_PARAMS["torque_filter_alpha"]
        # print(f"  setting torque_filter_alpha to {val}")
        # bus.write_torque_filter_alpha(joint_id, val)
        # time.sleep(DELAY_T)

        if STORE_TO_FLASH:
            print("  storing to flash...")
            bus.store_settings_to_flash(joint_id)
            time.sleep(0.2)

            # Step 2: Verify gear_ratio is still correct after flash store
            post_flash_gr = bus._read_parameter_f32(joint_id, recoil.Parameter.POSITION_CONTROLLER_GEAR_RATIO)
            if post_flash_gr != val:
                print(f"  WARNING: gear_ratio changed after flash store! Expected {val}, got {post_flash_gr}")
            else:
                print(f"  verified gear_ratio after flash store: {post_flash_gr}")

            # Step 3: Reload from flash and verify (simulates power cycle)
            print("  reloading from flash to verify persistence...")
            bus.load_settings_from_flash(joint_id)
            time.sleep(0.2)
            flash_gr = bus._read_parameter_f32(joint_id, recoil.Parameter.POSITION_CONTROLLER_GEAR_RATIO)
            if flash_gr != val:
                print(f"  ERROR: Flash persistence failed! Expected {val}, got {flash_gr}")
            else:
                print(f"  verified gear_ratio from flash: {flash_gr}")

        print()
        time.sleep(0.3)

    robot.stop()

    print("=" * 60)
    print("Configuration complete!")
    print()
    print("Next steps:")
    print("  1. Run read_all_configurations.py to verify gear_ratio=-15")
    print("  2. Run calibrate_electrical_offsets.py for each joint")
    print("  3. Run calibrate_joint_positions.py")
    print("  4. Test with test_actuator_interactive.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
