#!/usr/bin/env python3
"""
check_calibration.py

Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

Check if motor controllers have valid calibration data.
This script reads the flux offset and error status from each motor controller.
"""

import berkeley_humanoid_lite_lowlevel.recoil as recoil
from berkeley_humanoid_lite_lowlevel.robot import Humanoid


def check_motor_calibration(bus, device_id, joint_name):
    """Check calibration status for a single motor."""
    print(f"\nChecking {joint_name} (device {device_id}):")

    # Check if motor responds
    if not bus.ping(device_id):
        print(f"  ❌ ERROR: Motor not responding")
        return False

    # Read error status
    error_code = bus._read_parameter_u32(device_id, recoil.Parameter.ERROR)
    if error_code is None:
        print(f"  ❌ ERROR: Cannot read error status")
        return False

    # Check for calibration error
    if error_code & recoil.ErrorCode.CALIBRATION_ERROR:
        print(f"  ❌ CALIBRATION_ERROR flag is set (error code: 0x{error_code:04x})")
        return False
    elif error_code != recoil.ErrorCode.NO_ERROR:
        print(f"  ⚠️  Other errors present (error code: 0x{error_code:04x})")

    # Read flux offset
    flux_offset = bus.read_encoder_flux_offset(device_id)
    if flux_offset is None:
        print(f"  ❌ ERROR: Cannot read flux offset")
        return False

    print(f"  Flux offset: {flux_offset:.4f} radians")

    # Check if flux offset seems reasonable (typically should be between -π and π)
    if abs(flux_offset) > 6.3:  # slightly more than 2π to allow some margin
        print(f"  ⚠️  WARNING: Flux offset seems unusually large")

    # Check if flux offset is exactly zero (might indicate uncalibrated motor)
    if flux_offset == 0.0:
        print(f"  ⚠️  WARNING: Flux offset is exactly 0.0 - motor may not be calibrated")
        return False

    print(f"  ✅ Calibration appears valid")
    return True


def main():
    print("=" * 80)
    print("Motor Controller Calibration Check")
    print("=" * 80)

    robot = Humanoid()

    all_valid = True
    calibration_status = {}

    for entry in robot.joints:
        bus, device_id, joint_name = entry
        is_valid = check_motor_calibration(bus, device_id, joint_name)
        calibration_status[joint_name] = is_valid
        if not is_valid:
            all_valid = False

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)

    for joint_name, is_valid in calibration_status.items():
        status = "✅ VALID" if is_valid else "❌ INVALID/UNCALIBRATED"
        print(f"  {joint_name:30s}: {status}")

    print("\n" + "=" * 80)

    if all_valid:
        print("✅ All motors have valid calibration")
        print("\nYou can proceed with normal operation.")
    else:
        print("❌ Some motors need calibration")
        print("\nPlease run the calibration script:")
        print("  python3 ./scripts/motor/calibrate_electrical_offset.py --channel can0 --id <device_id>")
        print("\nFor each uncalibrated motor.")

    print("=" * 80)

    robot.stop()


if __name__ == "__main__":
    main()
