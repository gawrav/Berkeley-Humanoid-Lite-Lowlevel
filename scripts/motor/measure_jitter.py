# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Joint Jitter Measurement Script

Measures baseline position and velocity noise on all joints while the PD
controller holds the current pose. This isolates mechanical/sensor jitter
from policy-induced motion.

Procedure:
  1. Pose the robot manually before running (e.g. standing on gantry)
  2. Script reads current positions and switches to position mode,
     holding exactly where each joint is — no position commands sent
  3. Records position + velocity at 50 Hz for --duration seconds
  4. Reports per-joint jitter statistics

Usage:
    uv run scripts/motor/measure_jitter.py
    uv run scripts/motor/measure_jitter.py --duration 20
    uv run scripts/motor/measure_jitter.py --rate 200
    uv run scripts/motor/measure_jitter.py --save jitter_gantry.json
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot import Humanoid


JOINT_NAMES = [
    "L_hip_roll", "L_hip_yaw", "L_hip_pitch",
    "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_roll", "R_hip_yaw", "R_hip_pitch",
    "R_knee", "R_ankle_pitch", "R_ankle_roll",
]

NUM_JOINTS = len(JOINT_NAMES)


def enter_position_hold(robot):
    """Read current positions and switch to position mode, holding in place."""
    import berkeley_humanoid_lite_lowlevel.recoil as recoil

    # Read current positions
    robot.update_joints()
    hold_pos = robot.joint_position_measured.copy()

    print(f"  Current joint positions captured:")
    for j in range(NUM_JOINTS):
        print(f"    {JOINT_NAMES[j]:<16} {hold_pos[j]:>+8.4f} rad ({np.degrees(hold_pos[j]):>+8.2f}°)")

    # Set targets to current positions, then switch to position mode
    robot.joint_position_target[:] = hold_pos
    for entry in robot.joints:
        bus, device_id, _ = entry
        bus.feed(device_id)
        bus.set_mode(device_id, recoil.Mode.POSITION)

    # One update cycle to latch the targets
    robot.update_joints()

    print(f"  Position mode active — holding current pose.")
    return hold_pos


def record_holding(robot, duration, rate_hz, hold_pos):
    """Record position + velocity while PD controller holds the current pose."""
    rate = RateLimiter(frequency=rate_hz)
    steps = int(duration * rate_hz)

    target_pos = hold_pos

    timestamps = []
    positions = []
    velocities = []
    imu_quats = []
    imu_ang_vels = []

    print(f"  Recording {duration:.1f}s at {rate_hz:.0f} Hz ({steps} samples)...")

    t0 = time.perf_counter()

    for i in range(steps):
        # PD controller holds — target never changes from initial capture
        robot.update_joints()

        # Read IMU
        obs = robot.get_observations()
        quat = obs[0:4].copy()
        ang_vel = obs[4:7].copy()

        timestamps.append(time.perf_counter() - t0)
        positions.append(robot.joint_position_measured.copy())
        velocities.append(robot.joint_velocity_measured.copy())
        imu_quats.append(quat)
        imu_ang_vels.append(ang_vel)

        if (i + 1) % int(rate_hz) == 0:
            elapsed = i + 1
            print(f"    {elapsed / rate_hz:.0f}s...", end="", flush=True)

        rate.sleep()

    print(" done")

    return {
        "timestamps": np.array(timestamps),
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "imu_quats": np.array(imu_quats),
        "imu_ang_vels": np.array(imu_ang_vels),
        "target": target_pos,
    }


def analyze(data, rate_hz):
    """Compute and print jitter statistics."""
    positions = data["positions"]
    velocities = data["velocities"]
    target = data["target"]
    imu_ang_vels = data["imu_ang_vels"]
    imu_quats = data["imu_quats"]
    n_samples = len(positions)
    dt = 1.0 / rate_hz

    # --- Position jitter ---
    pos_mean = positions.mean(axis=0)
    pos_std = positions.std(axis=0)
    pos_ptp = positions.ptp(axis=0)  # peak-to-peak
    pos_error = positions - target  # deviation from hold position

    print(f"\n{'='*78}")
    print(f"  POSITION JITTER ({n_samples} samples, {n_samples * dt:.1f}s)")
    print(f"{'='*78}")
    print(f"  {'Joint':<16} {'Std (deg)':>10} {'PtP (deg)':>10} "
          f"{'Mean err':>10} {'Max |err|':>10}")
    print(f"  {'-'*58}")

    for j in range(NUM_JOINTS):
        std_deg = np.degrees(pos_std[j])
        ptp_deg = np.degrees(pos_ptp[j])
        me_deg = np.degrees(np.mean(pos_error[:, j]))
        maxe_deg = np.degrees(np.max(np.abs(pos_error[:, j])))
        flag = " <--" if std_deg > 0.5 else ""
        print(f"  {JOINT_NAMES[j]:<16} {std_deg:>10.3f} {ptp_deg:>10.3f} "
              f"{me_deg:>+10.3f} {maxe_deg:>10.3f}{flag}")

    # --- Velocity noise ---
    vel_mean = velocities.mean(axis=0)
    vel_std = velocities.std(axis=0)
    vel_rms = np.sqrt(np.mean(velocities**2, axis=0))
    vel_max = np.max(np.abs(velocities), axis=0)

    print(f"\n{'='*78}")
    print(f"  VELOCITY NOISE")
    print(f"{'='*78}")
    print(f"  {'Joint':<16} {'RMS (rad/s)':>12} {'Std (rad/s)':>12} "
          f"{'Max (rad/s)':>12}")
    print(f"  {'-'*54}")

    for j in range(NUM_JOINTS):
        flag = " <--" if vel_rms[j] > 1.0 else ""
        print(f"  {JOINT_NAMES[j]:<16} {vel_rms[j]:>12.4f} {vel_std[j]:>12.4f} "
              f"{vel_max[j]:>12.4f}{flag}")

    # --- Velocity jerk (smoothness) ---
    vel_diff = np.diff(velocities, axis=0) / dt
    jerk_rms = np.sqrt(np.mean(vel_diff**2, axis=0))

    print(f"\n{'='*78}")
    print(f"  VELOCITY JERK (smoothness, lower = smoother)")
    print(f"{'='*78}")
    print(f"  {'Joint':<16} {'RMS (rad/s²)':>14}")
    print(f"  {'-'*32}")

    for j in range(NUM_JOINTS):
        flag = " <--" if jerk_rms[j] > 50 else ""
        print(f"  {JOINT_NAMES[j]:<16} {jerk_rms[j]:>14.2f}{flag}")

    # --- Frequency analysis: dominant noise frequency per joint ---
    print(f"\n{'='*78}")
    print(f"  DOMINANT NOISE FREQUENCY (FFT of position signal)")
    print(f"{'='*78}")
    print(f"  {'Joint':<16} {'Freq (Hz)':>10} {'Amplitude':>10}")
    print(f"  {'-'*38}")

    freqs = np.fft.rfftfreq(n_samples, d=dt)

    for j in range(NUM_JOINTS):
        signal = positions[:, j] - pos_mean[j]
        fft_mag = np.abs(np.fft.rfft(signal))
        # Skip DC (index 0)
        fft_mag[0] = 0
        peak_idx = np.argmax(fft_mag)
        peak_freq = freqs[peak_idx]
        peak_amp = fft_mag[peak_idx] * 2 / n_samples  # amplitude in rad
        peak_amp_deg = np.degrees(peak_amp)
        print(f"  {JOINT_NAMES[j]:<16} {peak_freq:>10.2f} {peak_amp_deg:>9.4f}°")

    # --- IMU stability ---
    grav_xy = np.sqrt(imu_quats[:, 1]**2 + imu_quats[:, 2]**2)
    tilt_deg = np.degrees(np.arcsin(np.clip(grav_xy, -1, 1)) * 2)

    ang_vel_rms = np.sqrt(np.mean(imu_ang_vels**2, axis=0))

    print(f"\n{'='*78}")
    print(f"  IMU STABILITY")
    print(f"{'='*78}")
    print(f"  Angular velocity RMS: [{ang_vel_rms[0]:.4f}, {ang_vel_rms[1]:.4f}, "
          f"{ang_vel_rms[2]:.4f}] rad/s")
    print(f"  Quaternion std:       [{imu_quats.std(axis=0)[0]:.5f}, "
          f"{imu_quats.std(axis=0)[1]:.5f}, {imu_quats.std(axis=0)[2]:.5f}, "
          f"{imu_quats.std(axis=0)[3]:.5f}]")

    # --- Left vs Right symmetry ---
    print(f"\n{'='*78}")
    print(f"  LEFT vs RIGHT SYMMETRY (position std ratio)")
    print(f"{'='*78}")
    print(f"  {'Joint pair':<30} {'L std (deg)':>12} {'R std (deg)':>12} {'Ratio L/R':>10}")
    print(f"  {'-'*66}")

    pairs = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]
    pair_names = ["hip_roll", "hip_yaw", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]

    for (li, ri), name in zip(pairs, pair_names):
        l_std = np.degrees(pos_std[li])
        r_std = np.degrees(pos_std[ri])
        ratio = l_std / r_std if r_std > 1e-6 else float("inf")
        flag = " <--" if ratio > 2 or ratio < 0.5 else ""
        print(f"  {name:<30} {l_std:>12.4f} {r_std:>12.4f} {ratio:>10.2f}{flag}")

    # --- Summary ---
    worst_pos = np.argmax(pos_std)
    worst_vel = np.argmax(vel_rms)

    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"  Noisiest joint (position): {JOINT_NAMES[worst_pos]} "
          f"(std={np.degrees(pos_std[worst_pos]):.3f}°, "
          f"peak-to-peak={np.degrees(pos_ptp[worst_pos]):.3f}°)")
    print(f"  Noisiest joint (velocity): {JOINT_NAMES[worst_vel]} "
          f"(RMS={vel_rms[worst_vel]:.4f} rad/s)")
    print(f"  Overall position std: {np.degrees(pos_std.mean()):.3f}° mean, "
          f"{np.degrees(pos_std.max()):.3f}° worst")
    print(f"  Overall velocity RMS: {vel_rms.mean():.4f} rad/s mean, "
          f"{vel_rms.max():.4f} rad/s worst")
    print(f"{'='*78}")


def save_results(data, path, args):
    """Save raw recording data to JSON for later analysis."""
    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "duration": args.duration,
            "rate_hz": args.rate,
            "n_samples": len(data["timestamps"]),
            "joint_names": JOINT_NAMES,
        },
        "target_positions": data["target"].tolist(),
        "timestamps": data["timestamps"].tolist(),
        "positions": data["positions"].tolist(),
        "velocities": data["velocities"].tolist(),
        "imu_quaternions": data["imu_quats"].tolist(),
        "imu_angular_velocities": data["imu_ang_vels"].tolist(),
    }

    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Saved raw data to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure baseline joint jitter while holding standing pose")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Recording duration in seconds (default: 10)")
    parser.add_argument("--rate", type=float, default=50.0,
                        help="Sampling rate in Hz (default: 50, matching policy rate)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save raw data to JSON file")
    args = parser.parse_args()

    print(f"\n{'='*78}")
    print(f"  Joint Jitter Measurement")
    print(f"  Duration: {args.duration}s  Rate: {args.rate} Hz")
    print(f"{'='*78}\n")

    print("  Initializing robot...")
    robot = Humanoid()
    robot.enter_damping()

    try:
        # Capture current pose and switch to position hold
        hold_pos = enter_position_hold(robot)

        # Record
        data = record_holding(robot, args.duration, args.rate, hold_pos)

        # Analyze
        analyze(data, args.rate)

        if args.save:
            save_results(data, args.save, args)

    except KeyboardInterrupt:
        print("\n\n  Interrupted by Ctrl+C")

    finally:
        print("\n  Returning to damping mode...")
        robot.stop()


if __name__ == "__main__":
    main()
