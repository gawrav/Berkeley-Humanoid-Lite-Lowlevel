# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Show Joint Positions

Displays current joint positions vs init (standing) positions.
Useful for verifying robot pose before running locomotion.

Usage:
    uv run scripts/show_joint_positions.py
"""

import time
import numpy as np

from berkeley_humanoid_lite_lowlevel.robot import Humanoid


JOINT_NAMES = [
    "L_hip_roll",
    "L_hip_yaw",
    "L_hip_pitch",
    "L_knee",
    "L_ankle_pitch",
    "L_ankle_roll",
    "R_hip_roll",
    "R_hip_yaw",
    "R_hip_pitch",
    "R_knee",
    "R_ankle_pitch",
    "R_ankle_roll",
]


def main():
    print("\nInitializing robot...")
    robot = Humanoid()
    robot.enter_damping()

    print("Reading joint positions (Ctrl+C to stop)...\n")

    init_positions = robot.rl_init_positions

    try:
        while True:
            # Update joint readings
            robot.update_joints()
            current = robot.joint_position_measured

            # Clear screen
            print("\033[H\033[J", end="")

            print("=" * 85)
            print(f"{'JOINT POSITIONS':^85}")
            print("=" * 85)
            print()
            print(f"  {'Joint':<15} {'Current (rad)':>14} {'Init (rad)':>14} {'Delta (rad)':>12} {'Delta (deg)':>12}")
            print("  " + "-" * 75)

            for i, name in enumerate(JOINT_NAMES):
                cur = current[i]
                init = init_positions[i]
                delta = init - cur
                delta_deg = np.rad2deg(delta)

                # Warning for large differences
                if abs(delta_deg) > 20:
                    warn = " << LARGE"
                elif abs(delta_deg) > 10:
                    warn = " <<"
                else:
                    warn = ""

                print(f"  {name:<15} {cur:>+14.4f} {init:>+14.4f} {delta:>+12.4f} {delta_deg:>+12.1f}°{warn}")

            print()
            print("  " + "-" * 75)
            print("  Delta = Init - Current (positive = joint needs to move in + direction)")
            print()
            print("=" * 85)
            print("Press Ctrl+C to stop")

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    robot.stop()
    print("Done.")


if __name__ == "__main__":
    main()
