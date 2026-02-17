# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Interactive Joint Test via Humanoid Class

This script tests individual joints through the Humanoid class code path,
ensuring joint_axis_directions and position_offsets are applied correctly.
This is the same code path used by run_locomotion.py.

Usage:
    uv run scripts/test_joint_humanoid.py

Controls:
    0-9, a, b : Select joint (0-11)
    +/=       : Move selected joint +0.05 rad
    -/_       : Move selected joint -0.05 rad
    r         : Read all joint positions
    z         : Zero selected joint to current position
    d         : Enable position control (from DAMPING)
    q         : Quit
"""

import sys
import select
import termios
import tty
import time

import numpy as np
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot import Humanoid, State
import berkeley_humanoid_lite_lowlevel.recoil as recoil


STEP_SIZE = 0.05  # radians - smaller step for safety

JOINT_NAMES = [
    "left_hip_roll",      # 0
    "left_hip_yaw",       # 1
    "left_hip_pitch",     # 2
    "left_knee_pitch",    # 3
    "left_ankle_pitch",   # 4
    "left_ankle_roll",    # 5
    "right_hip_roll",     # 6
    "right_hip_yaw",      # 7
    "right_hip_pitch",    # 8
    "right_knee_pitch",   # 9
    "right_ankle_pitch",  # 10 (a)
    "right_ankle_roll",   # 11 (b)
]


def get_key_nonblocking():
    """Get a key press without blocking. Returns None if no key pressed."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def print_help():
    print("\n=== Interactive Joint Test (via Humanoid) ===")
    print("This tests through the same code path as run_locomotion.py")
    print("\nJoint Selection:")
    for i, name in enumerate(JOINT_NAMES):
        key = str(i) if i < 10 else chr(ord('a') + i - 10)
        print(f"  {key} : [{i:2d}] {name}")
    print("\nCommands:")
    print("  d     : Enable position control (from DAMPING)")
    print("  +/=   : Move selected joint +0.05 rad")
    print("  -/_   : Move selected joint -0.05 rad")
    print("  r     : Read all joint positions")
    print("  z     : Zero selected joint to current measured position")
    print("  q     : Quit")
    print("  h     : Show this help")
    print("==============================================\n")


def main():
    print("\nInitializing Humanoid (this will connect to all joints)...")

    robot = Humanoid()

    print("\nAll joints connected!")
    print_help()

    # Enter damping mode (safe start)
    robot.enter_damping()

    # Save terminal settings
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        # Set terminal to raw mode for non-blocking key input
        tty.setcbreak(sys.stdin.fileno())

        rate = RateLimiter(frequency=50.0)

        selected_joint = 0
        joint_targets = np.zeros(12, dtype=np.float32)
        mode_enabled = False
        last_print_time = 0

        print(f"\nSelected joint: [{selected_joint}] {JOINT_NAMES[selected_joint]}")
        print("Press 'd' to enable position control, 'h' for help")

        while True:
            # Check for key press
            key = get_key_nonblocking()

            if key:
                # Joint selection (0-9, a, b)
                if key in '0123456789':
                    selected_joint = int(key)
                    print(f"\nSelected: [{selected_joint}] {JOINT_NAMES[selected_joint]}")
                    print(f"  Measured: {robot.joint_position_measured[selected_joint]:+.3f} rad")
                    if mode_enabled:
                        print(f"  Target:   {joint_targets[selected_joint]:+.3f} rad")

                elif key == 'a':
                    selected_joint = 10
                    print(f"\nSelected: [{selected_joint}] {JOINT_NAMES[selected_joint]}")
                    print(f"  Measured: {robot.joint_position_measured[selected_joint]:+.3f} rad")

                elif key == 'b':
                    selected_joint = 11
                    print(f"\nSelected: [{selected_joint}] {JOINT_NAMES[selected_joint]}")
                    print(f"  Measured: {robot.joint_position_measured[selected_joint]:+.3f} rad")

                elif key == 'q':
                    print("\n\nQuitting...")
                    break

                elif key == 'd':
                    if not mode_enabled:
                        # Initialize targets to current measured positions
                        joint_targets[:] = robot.joint_position_measured[:]

                        # Switch all joints to POSITION mode
                        for entry in robot.joints:
                            bus, device_id, _ = entry
                            bus.feed(device_id)
                            bus.set_mode(device_id, recoil.Mode.POSITION)

                        mode_enabled = True
                        print(f"\nPosition control ENABLED")
                        print("Joint targets initialized to current positions")
                    else:
                        print("\nAlready enabled.")

                elif key == 'h':
                    print_help()

                elif key in ['+', '=']:
                    if mode_enabled:
                        joint_targets[selected_joint] += STEP_SIZE
                        print(f"\n[{selected_joint}] {JOINT_NAMES[selected_joint]}: target = {joint_targets[selected_joint]:+.3f} rad (+{STEP_SIZE})")
                    else:
                        print("\nPress 'd' first to enable control")

                elif key in ['-', '_']:
                    if mode_enabled:
                        joint_targets[selected_joint] -= STEP_SIZE
                        print(f"\n[{selected_joint}] {JOINT_NAMES[selected_joint]}: target = {joint_targets[selected_joint]:+.3f} rad (-{STEP_SIZE})")
                    else:
                        print("\nPress 'd' first to enable control")

                elif key == 'z':
                    if mode_enabled:
                        joint_targets[selected_joint] = robot.joint_position_measured[selected_joint]
                        print(f"\n[{selected_joint}] {JOINT_NAMES[selected_joint]}: target zeroed to {joint_targets[selected_joint]:+.3f} rad")
                    else:
                        print("\nPress 'd' first to enable control")

                elif key == 'r':
                    print("\n--- All Joint Positions ---")
                    for i, name in enumerate(JOINT_NAMES):
                        marker = " *" if i == selected_joint else ""
                        measured = robot.joint_position_measured[i]
                        if mode_enabled:
                            target = joint_targets[i]
                            error = target - measured
                            print(f"  [{i:2d}] {name:20s}: meas={measured:+.3f}  tgt={target:+.3f}  err={error:+.3f}{marker}")
                        else:
                            print(f"  [{i:2d}] {name:20s}: meas={measured:+.3f}{marker}")
                    print("---------------------------")

            # Update joints through Humanoid class
            if mode_enabled:
                robot.joint_position_target[:] = joint_targets[:]
            else:
                robot.joint_position_target[:] = robot.joint_position_measured[:]

            robot.update_joints()

            # Print status periodically
            current_time = time.time()
            if current_time - last_print_time > 0.5:
                status = "POSITION" if mode_enabled else "DAMPING"
                j = selected_joint
                meas = robot.joint_position_measured[j]
                sys.stdout.write(f"\r[{status}] Joint {j} ({JOINT_NAMES[j]}): {meas:+.3f} rad  ")
                if mode_enabled:
                    tgt = joint_targets[j]
                    sys.stdout.write(f"Target: {tgt:+.3f} rad  ")
                sys.stdout.flush()
                last_print_time = current_time

            rate.sleep()

    except KeyboardInterrupt:
        print("\n\nInterrupted by Ctrl+C")

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # Stop robot (enters damping then idle)
        print("\nStopping robot...")
        robot.stop()


if __name__ == "__main__":
    main()
