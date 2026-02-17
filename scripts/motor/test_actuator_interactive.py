# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Interactive Actuator Test Script

This script allows you to test individual actuators safely before running the full robot.
It provides keyboard-based interactive control to read positions and command small movements.

Usage:
    uv run scripts/motor/test_actuator_interactive.py -c can1 -i 1

Joint ID Reference:
    can1: left_hip_roll(1), left_hip_yaw(3), left_hip_pitch(5),
          left_knee_pitch(7), left_ankle_pitch(11), left_ankle_roll(13)
    can2: right_hip_roll(2), right_hip_yaw(4), right_hip_pitch(6),
          right_knee_pitch(8), right_ankle_pitch(12), right_ankle_roll(14)
"""

import sys
import select
import termios
import tty
import time

from loop_rate_limiters import RateLimiter
import berkeley_humanoid_lite_lowlevel.recoil as recoil


# Moderate settings - balanced stiffness and damping
# High KD prevents wobble, moderate KP/torque for smooth movement
KP = 10.0
KD = 2.0
TORQUE_LIMIT = 3.0
STEP_SIZE = 0.1  # radians


def get_key_nonblocking():
    """Get a key press without blocking. Returns None if no key pressed."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def print_help():
    print("\n=== Interactive Actuator Test ===")
    print("Commands:")
    print("  d     : Enable position control (from DAMPING)")
    print("  +/=   : Move +0.1 rad")
    print("  -/_   : Move -0.1 rad")
    print("  r     : Read and print current position")
    print("  z     : Zero target to current measured position")
    print("  q     : Quit (return to IDLE)")
    print("  h     : Show this help")
    print("================================\n")


def main():
    args = recoil.util.get_args()
    bus = recoil.Bus(channel=args.channel, bitrate=1000000)
    device_id = args.id

    print(f"\nConnecting to actuator ID {device_id} on {args.channel}...")

    # Check connection
    if not bus.ping(device_id):
        print(f"ERROR: Actuator {device_id} not responding on {args.channel}")
        bus.stop()
        return

    print(f"Actuator {device_id} is online!")

    # Configure safe settings
    print(f"\nConfiguring safe settings: kp={KP}, kd={KD}, torque_limit={TORQUE_LIMIT}")
    bus.write_position_kp(device_id, KP)
    time.sleep(0.001)
    bus.write_position_kd(device_id, KD)
    time.sleep(0.001)
    bus.write_torque_limit(device_id, TORQUE_LIMIT)
    time.sleep(0.001)

    # Start in DAMPING mode for safety
    bus.set_mode(device_id, recoil.Mode.DAMPING)
    bus.feed(device_id)

    print_help()
    print("Starting in DAMPING mode. Press 'd' to enable position control.")

    # Save terminal settings
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        # Set terminal to raw mode for non-blocking key input
        tty.setcbreak(sys.stdin.fileno())

        rate = RateLimiter(frequency=50.0)

        target_position = 0.0
        measured_position = 0.0
        measured_velocity = 0.0
        mode_enabled = False
        last_print_time = 0

        while True:
            # Check for key press
            key = get_key_nonblocking()

            if key:
                if key == 'q':
                    print("\n\nQuitting...")
                    break

                elif key == 'd':
                    if not mode_enabled:
                        # Read current position first
                        pos, vel = bus.write_read_pdo_2(device_id, 0, 0)
                        if pos is not None:
                            target_position = pos
                            measured_position = pos

                        bus.set_mode(device_id, recoil.Mode.POSITION)
                        bus.feed(device_id)
                        mode_enabled = True
                        print(f"\nPosition control ENABLED. Target: {target_position:.3f} rad")
                    else:
                        print("\nAlready enabled.")

                elif key == 'h':
                    print_help()

                elif key in ['+', '=']:
                    if mode_enabled:
                        target_position += STEP_SIZE
                        print(f"\nTarget: {target_position:.3f} rad (+{STEP_SIZE})")
                    else:
                        print("\nPress 'd' first to enable control")

                elif key in ['-', '_']:
                    if mode_enabled:
                        target_position -= STEP_SIZE
                        print(f"\nTarget: {target_position:.3f} rad (-{STEP_SIZE})")
                    else:
                        print("\nPress 'd' first to enable control")

                elif key == 'z':
                    if mode_enabled:
                        target_position = measured_position
                        print(f"\nTarget zeroed to measured: {target_position:.3f} rad")
                    else:
                        print("\nPress 'd' first to enable control")

                elif key == 'r':
                    print(f"\nMeasured position: {measured_position:.4f} rad")
                    print(f"Measured velocity: {measured_velocity:.4f} rad/s")
                    if mode_enabled:
                        print(f"Target position:   {target_position:.4f} rad")
                        print(f"Position error:    {target_position - measured_position:.4f} rad")

            # Send command and read feedback
            if mode_enabled:
                pos, vel = bus.write_read_pdo_2(device_id, target_position, 0.0)
            else:
                # In damping mode, just read position
                pos, vel = bus.write_read_pdo_2(device_id, 0, 0)

            if pos is not None:
                measured_position = pos
            if vel is not None:
                measured_velocity = vel

            # Print status periodically (every 0.5 seconds)
            current_time = time.time()
            if current_time - last_print_time > 0.5:
                status = "POSITION" if mode_enabled else "DAMPING"
                sys.stdout.write(f"\r[{status}] Pos: {measured_position:+.3f} rad  Vel: {measured_velocity:+.3f} rad/s  ")
                if mode_enabled:
                    sys.stdout.write(f"Target: {target_position:+.3f} rad  ")
                sys.stdout.flush()
                last_print_time = current_time

            rate.sleep()

    except KeyboardInterrupt:
        print("\n\nInterrupted by Ctrl+C")

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # Return to safe mode
        print("Returning to IDLE mode...")
        bus.set_mode(device_id, recoil.Mode.IDLE)
        bus.stop()
        print("Done.")


if __name__ == "__main__":
    main()
