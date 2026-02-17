# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
calibrate_standing.py

Simple calibration script - position the robot in a standing upright pose
(all joints straight/neutral) and run this script. It will save the current
encoder readings as offsets so that standing = all zeros.

Usage:
1. Position robot standing upright with legs straight
2. Run this script
3. Press Y (or button 3) to save calibration
"""

import time
import numpy as np
import yaml

from berkeley_humanoid_lite_lowlevel.robot import Humanoid


robot = Humanoid()

joint_names = [
    "left_hip_roll",
    "left_hip_yaw",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_roll",
    "right_hip_yaw",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
]

# Use the same directions as humanoid.py
joint_axis_directions = robot.joint_axis_directions

# Standing upright = all joints at 0
ideal_values = np.zeros(12)

print("=" * 50)
print("STANDING POSE CALIBRATION")
print("=" * 50)
print("\nPosition the robot in a standing upright pose:")
print("  - Legs straight (knees not bent)")
print("  - Feet flat and parallel")
print("  - Hips neutral (not rotated)")
print("\nPress Y (button 3) when ready to save calibration.")
print("=" * 50)

# Read current positions
while robot.command_controller.commands.get("mode_switch") != 1:
    readings = np.array([joint[0].read_position_measured(joint[1]) for joint in robot.joints]) * joint_axis_directions

    print("\nCurrent readings (should be stable when robot is still):")
    for i, name in enumerate(joint_names):
        print(f"  {i:2d}. {name:20s}: {readings[i]:8.4f} rad ({np.rad2deg(readings[i]):7.2f} deg)")

    time.sleep(0.5)

# Calculate offsets: offset = reading - ideal (where ideal is 0)
final_readings = np.array([joint[0].read_position_measured(joint[1]) for joint in robot.joints]) * joint_axis_directions
offsets = final_readings - ideal_values

print("\n" + "=" * 50)
print("CALIBRATION SAVED")
print("=" * 50)
print("\nOffsets (raw_reading * direction - 0):")
for i, name in enumerate(joint_names):
    print(f"  {i:2d}. {name:20s}: {offsets[i]:8.4f} rad")

calibration_data = {
    "position_offsets": [float(offset) for offset in offsets],
}

with open("calibration.yaml", "w") as f:
    yaml.dump(calibration_data, f)

print("\nSaved to calibration.yaml")

robot.stop()
