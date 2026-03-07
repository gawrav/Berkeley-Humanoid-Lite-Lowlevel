# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Sine Wave Kinematic Chain Test

Commands slow sine waves to joints on a SUSPENDED robot and measures tracking
performance: phase lag, amplitude ratio, and RMS tracking error. Uses the
Humanoid class code path (same as run_locomotion.py) so joint_axis_directions
and position_offsets are applied.

WARNING: The robot must be suspended (feet off the ground) before running!

Usage:
    # All joints together (default 0.5 Hz, 0.15 rad amplitude)
    uv run scripts/test_sine_kinematic.py --joints all

    # Single joint
    uv run scripts/test_sine_kinematic.py --joints 3

    # Set of joints (by index)
    uv run scripts/test_sine_kinematic.py --joints 0,1,2,6,7,8

    # Predefined groups
    uv run scripts/test_sine_kinematic.py --joints left_leg
    uv run scripts/test_sine_kinematic.py --joints right_leg
    uv run scripts/test_sine_kinematic.py --joints hips
    uv run scripts/test_sine_kinematic.py --joints knees
    uv run scripts/test_sine_kinematic.py --joints ankles

    # Custom parameters
    uv run scripts/test_sine_kinematic.py --joints all --freq 1.0 --amplitude 0.1 --duration 10

    # Save raw data to CSV
    uv run scripts/test_sine_kinematic.py --joints all --save sine_test.csv

Joint Index Reference:
    0: left_hip_roll       6: right_hip_roll
    1: left_hip_yaw        7: right_hip_yaw
    2: left_hip_pitch      8: right_hip_pitch
    3: left_knee_pitch     9: right_knee_pitch
    4: left_ankle_pitch   10: right_ankle_pitch
    5: left_ankle_roll    11: right_ankle_roll
"""

import argparse
import sys
import time

import numpy as np
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot import Humanoid
import berkeley_humanoid_lite_lowlevel.recoil as recoil


CONTROL_RATE = 50.0  # Hz — matches test_joint_humanoid.py

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
    "right_ankle_pitch",  # 10
    "right_ankle_roll",   # 11
]

JOINT_GROUPS = {
    "all":        list(range(12)),
    "left_leg":   [0, 1, 2, 3, 4, 5],
    "right_leg":  [6, 7, 8, 9, 10, 11],
    "hips":       [0, 1, 2, 6, 7, 8],
    "knees":      [3, 9],
    "ankles":     [4, 5, 10, 11],
    "hip_rolls":  [0, 6],
    "hip_yaws":   [1, 7],
    "hip_pitches": [2, 8],
}


def parse_joints(joints_str):
    """Parse --joints argument into a list of joint indices."""
    if joints_str in JOINT_GROUPS:
        return JOINT_GROUPS[joints_str]

    # Try comma-separated indices
    try:
        indices = [int(x.strip()) for x in joints_str.split(",")]
        for idx in indices:
            if idx < 0 or idx >= 12:
                print(f"ERROR: Joint index {idx} out of range (0-11)")
                sys.exit(1)
        return indices
    except ValueError:
        print(f"ERROR: Unknown joint specifier '{joints_str}'")
        print(f"  Use: all, left_leg, right_leg, hips, knees, ankles,")
        print(f"       hip_rolls, hip_yaws, hip_pitches,")
        print(f"       or comma-separated indices (e.g., 0,1,2)")
        sys.exit(1)


def analyze_phase_lag(commanded, measured, sample_rate):
    """Compute phase lag between commanded and measured signals using cross-correlation.

    Returns:
        lag_ms: Phase lag in milliseconds
        lag_deg: Phase lag in degrees (relative to the sine frequency)
        amplitude_ratio: Measured amplitude / commanded amplitude
        rms_error: RMS tracking error in radians
    """
    # Remove mean (center around zero)
    cmd = commanded - np.mean(commanded)
    meas = measured - np.mean(measured)

    # Cross-correlation to find lag
    correlation = np.correlate(meas, cmd, mode='full')
    lags = np.arange(-len(cmd) + 1, len(cmd))

    # Search only positive lags (response after command), up to 200ms
    max_lag_samples = int(0.2 * sample_rate)
    positive_mask = (lags >= 0) & (lags <= max_lag_samples)
    positive_lags = lags[positive_mask]
    positive_corr = correlation[positive_mask]

    if len(positive_corr) == 0:
        return 0.0, 0.0, 0.0, 0.0

    peak_idx = np.argmax(positive_corr)
    lag_samples = positive_lags[peak_idx]
    lag_ms = lag_samples / sample_rate * 1000.0

    # Amplitude ratio (RMS-based)
    cmd_amp = np.sqrt(2) * np.std(cmd)  # RMS to amplitude for sine
    meas_amp = np.sqrt(2) * np.std(meas)
    amplitude_ratio = meas_amp / cmd_amp if cmd_amp > 1e-6 else 0.0

    # RMS tracking error
    rms_error = np.sqrt(np.mean((commanded - measured) ** 2))

    return lag_ms, lag_samples, amplitude_ratio, rms_error


def run_sine_test(robot, test_joints, freq, amplitude, duration, settle_time=2.0):
    """Run sine wave test on specified joints.

    Args:
        robot: Humanoid instance (already in DAMPING mode)
        test_joints: List of joint indices to excite
        freq: Sine frequency in Hz
        amplitude: Sine amplitude in radians
        duration: Test duration in seconds
        settle_time: Time to hold at center before starting sine

    Returns:
        results: Dict with per-joint analysis
        raw_data: Dict with timestamps, commanded, measured arrays
    """
    rate = RateLimiter(frequency=CONTROL_RATE)

    # Read current positions while still in DAMPING mode
    # Do a few update cycles to populate joint_position_measured
    for _ in range(5):
        robot.joint_position_target[:] = 0.0  # Doesn't matter in DAMPING mode
        robot.update_joints()
        rate.sleep()

    center_positions = robot.joint_position_measured.copy()

    # Now switch to POSITION mode with targets set to current positions
    robot.joint_position_target[:] = center_positions
    for entry in robot.joints:
        bus, device_id, _ = entry
        bus.feed(device_id)
        bus.set_mode(device_id, recoil.Mode.POSITION)

    print(f"\n  Center positions (current):")
    for j in test_joints:
        print(f"    [{j:2d}] {JOINT_NAMES[j]:20s}: {center_positions[j]:+.3f} rad")

    # Settle at center
    print(f"\n  Settling at center for {settle_time:.1f}s...", end="", flush=True)
    settle_steps = int(settle_time * CONTROL_RATE)
    for _ in range(settle_steps):
        robot.joint_position_target[:] = center_positions
        robot.update_joints()
        rate.sleep()
    print(" done")

    # Run sine wave
    total_steps = int(duration * CONTROL_RATE)
    timestamps = np.zeros(total_steps)
    commanded = np.zeros((total_steps, 12))
    measured = np.zeros((total_steps, 12))

    print(f"  Running sine wave ({freq} Hz, ±{amplitude} rad, {duration}s)...", end="", flush=True)
    t0 = time.perf_counter()

    for i in range(total_steps):
        t = i / CONTROL_RATE
        sine_val = amplitude * np.sin(2 * np.pi * freq * t)

        # Set targets: sine on test joints, hold center on others
        targets = center_positions.copy()
        for j in test_joints:
            targets[j] = center_positions[j] + sine_val

        robot.joint_position_target[:] = targets
        robot.update_joints()

        timestamps[i] = time.perf_counter() - t0
        commanded[i, :] = targets
        measured[i, :] = robot.joint_position_measured.copy()

        rate.sleep()

        # Progress every second
        if (i + 1) % int(CONTROL_RATE) == 0:
            print(f" {(i+1)/CONTROL_RATE:.0f}s", end="", flush=True)

    print(" done")

    # Return to center and settle
    print("  Returning to center...", end="", flush=True)
    for _ in range(int(1.0 * CONTROL_RATE)):
        robot.joint_position_target[:] = center_positions
        robot.update_joints()
        rate.sleep()
    print(" done")

    # Analyze each test joint
    # Skip the first cycle to avoid transient
    skip_samples = int(CONTROL_RATE / freq) if freq > 0 else 0
    if skip_samples >= total_steps:
        skip_samples = 0

    results = {}
    for j in test_joints:
        cmd_signal = commanded[skip_samples:, j]
        meas_signal = measured[skip_samples:, j]

        lag_ms, lag_samples, amp_ratio, rms_err = analyze_phase_lag(
            cmd_signal, meas_signal, CONTROL_RATE
        )

        # Convert lag to degrees at the test frequency
        lag_deg = lag_ms / 1000.0 * freq * 360.0

        results[j] = {
            "lag_ms": lag_ms,
            "lag_samples": lag_samples,
            "lag_deg": lag_deg,
            "amplitude_ratio": amp_ratio,
            "rms_error_rad": rms_err,
            "rms_error_deg": np.rad2deg(rms_err),
        }

    raw_data = {
        "timestamps": timestamps,
        "commanded": commanded,
        "measured": measured,
    }

    return results, raw_data


def print_results(results, freq):
    """Print analysis results in a formatted table."""
    print(f"\n{'='*72}")
    print(f"Sine Wave Tracking Results (f = {freq} Hz)")
    print(f"{'='*72}")
    print(f"  {'Joint':<20s} {'Lag (ms)':>8s} {'Lag (°)':>8s} {'Amp Ratio':>10s} {'RMS Err':>10s}")
    print(f"  {'-'*60}")

    lags = []
    for j in sorted(results.keys()):
        r = results[j]
        lags.append(r["lag_ms"])
        print(
            f"  [{j:2d}] {JOINT_NAMES[j]:<16s}"
            f" {r['lag_ms']:>7.1f}"
            f" {r['lag_deg']:>7.1f}"
            f" {r['amplitude_ratio']:>9.3f}"
            f" {r['rms_error_deg']:>8.2f}°"
        )

    print(f"  {'-'*60}")
    lags = np.array(lags)
    print(f"  {'Mean lag:':<20s} {np.mean(lags):>7.1f} ms")
    print(f"  {'Max lag:':<20s} {np.max(lags):>7.1f} ms")
    print(f"  {'Std lag:':<20s} {np.std(lags):>7.1f} ms")
    print(f"{'='*72}")


def save_csv(filepath, raw_data, test_joints):
    """Save raw commanded and measured data to CSV."""
    timestamps = raw_data["timestamps"]
    commanded = raw_data["commanded"]
    measured = raw_data["measured"]

    header_parts = ["timestamp_s"]
    for j in test_joints:
        header_parts.append(f"cmd_{JOINT_NAMES[j]}")
        header_parts.append(f"meas_{JOINT_NAMES[j]}")
    header = ",".join(header_parts)

    rows = []
    for i in range(len(timestamps)):
        parts = [f"{timestamps[i]:.6f}"]
        for j in test_joints:
            parts.append(f"{commanded[i, j]:.6f}")
            parts.append(f"{measured[i, j]:.6f}")
        rows.append(",".join(parts))

    with open(filepath, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")

    print(f"\nSaved {len(rows)} samples to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Sine wave kinematic chain test for suspended robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Joint groups: all, left_leg, right_leg, hips, knees, ankles,
              hip_rolls, hip_yaws, hip_pitches
Or comma-separated indices: 0,1,2 or single index: 3
        """,
    )
    parser.add_argument(
        "--joints", type=str, default="all",
        help="Joints to test (default: all). See groups above or use indices.",
    )
    parser.add_argument(
        "--freq", type=float, default=0.5,
        help="Sine frequency in Hz (default: 0.5)",
    )
    parser.add_argument(
        "--amplitude", type=float, default=0.15,
        help="Sine amplitude in radians (default: 0.15)",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Test duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--settle", type=float, default=2.0,
        help="Settle time at center before sine starts (default: 2.0)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save raw data to CSV file",
    )
    args = parser.parse_args()

    test_joints = parse_joints(args.joints)

    print(f"\n{'='*72}")
    print("Sine Wave Kinematic Chain Test")
    print(f"{'='*72}")
    print(f"  WARNING: Robot must be SUSPENDED (feet off ground)!")
    print(f"  Joints:    {args.joints} → {[JOINT_NAMES[j] for j in test_joints]}")
    print(f"  Frequency: {args.freq} Hz")
    print(f"  Amplitude: ±{args.amplitude} rad ({np.rad2deg(args.amplitude):.1f}°)")
    print(f"  Duration:  {args.duration} s")
    print(f"  Rate:      {CONTROL_RATE} Hz")
    print(f"{'='*72}")

    input("\nPress Enter to start (Ctrl+C to abort)...")

    print("\nInitializing Humanoid...")
    robot = Humanoid()
    robot.enter_damping()

    try:
        results, raw_data = run_sine_test(
            robot, test_joints, args.freq, args.amplitude, args.duration, args.settle
        )
        print_results(results, args.freq)

        if args.save:
            save_csv(args.save, raw_data, test_joints)

    except KeyboardInterrupt:
        print("\n\nInterrupted by Ctrl+C")

    finally:
        print("\nStopping robot...")
        robot.stop()


if __name__ == "__main__":
    main()
