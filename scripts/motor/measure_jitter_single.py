# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Single-Joint Jitter Measurement Script

Measures baseline position and velocity noise on a single joint while the PD
controller holds the current pose. Uses raw CAN bus (bypasses Humanoid class),
matching the same interface as measure_delay.py.

Procedure:
  1. Pose the joint manually before running
  2. Script reads current position, switches to position mode, holds in place
  3. Records position + velocity at --rate Hz for --duration seconds
  4. Reports jitter statistics

Usage:
    uv run scripts/motor/measure_jitter_single.py -c can2 -i 1
    uv run scripts/motor/measure_jitter_single.py -c can0 -i 2 --duration 20
    uv run scripts/motor/measure_jitter_single.py -c can2 -i 1 --kp 20 --kd 2 --torque-limit 4
    uv run scripts/motor/measure_jitter_single.py -c can2 -i 1 --save jitter_L_hip_roll.json

Joint ID Reference:
    can2: left_hip_roll(1), left_hip_yaw(3), left_hip_pitch(5),
          left_knee_pitch(7), left_ankle_pitch(11), left_ankle_roll(13)
    can0: right_hip_roll(2), right_hip_yaw(4), right_hip_pitch(6),
          right_knee_pitch(8), right_ankle_pitch(12), right_ankle_roll(14)
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np
from loop_rate_limiters import RateLimiter

import berkeley_humanoid_lite_lowlevel.recoil as recoil


# Default PD gains (matching measure_delay.py)
DEFAULT_KP = 10.0
DEFAULT_KD = 2.0
DEFAULT_TORQUE_LIMIT = 3.0
DEFAULT_RATE = 200.0
DEFAULT_DURATION = 10.0


def setup_actuator(bus, device_id, kp, kd, torque_limit):
    """Configure actuator and verify connection."""
    if not bus.ping(device_id):
        print(f"ERROR: Actuator {device_id} not responding")
        return None

    print(f"  Actuator {device_id} online.")
    print(f"  Configuring: kp={kp}, kd={kd}, torque_limit={torque_limit}")

    bus.write_position_kp(device_id, kp)
    time.sleep(0.001)
    bus.write_position_kd(device_id, kd)
    time.sleep(0.001)
    bus.write_torque_limit(device_id, torque_limit)
    time.sleep(0.001)

    # Start in damping mode, read current position
    bus.set_mode(device_id, recoil.Mode.DAMPING)
    bus.feed(device_id)
    time.sleep(0.1)

    pos, vel = bus.write_read_pdo_2(device_id, 0, 0)
    if pos is None:
        print("  ERROR: Cannot read initial position")
        return None

    print(f"  Current position: {pos:.4f} rad ({np.degrees(pos):.2f}°)")
    return pos


def record_holding(bus, device_id, hold_pos, duration, rate_hz):
    """Switch to position mode, hold current position, record noise."""
    # Switch to position mode holding current position
    bus.set_mode(device_id, recoil.Mode.POSITION)
    bus.feed(device_id)

    # One cycle to latch the target
    bus.write_read_pdo_2(device_id, hold_pos, 0.0)
    bus.feed(device_id)
    time.sleep(0.05)

    rate = RateLimiter(frequency=rate_hz)
    steps = int(duration * rate_hz)

    timestamps = []
    positions = []
    velocities = []

    print(f"  Recording {duration:.1f}s at {rate_hz:.0f} Hz ({steps} samples)...")

    t0 = time.perf_counter()

    for i in range(steps):
        bus.feed(device_id)
        pos, vel = bus.write_read_pdo_2(device_id, hold_pos, 0.0)

        timestamps.append(time.perf_counter() - t0)
        if pos is not None:
            positions.append(pos)
        else:
            positions.append(positions[-1] if positions else hold_pos)
        if vel is not None:
            velocities.append(vel)
        else:
            velocities.append(velocities[-1] if velocities else 0.0)

        if (i + 1) % int(rate_hz) == 0:
            print(f"    {(i + 1) / rate_hz:.0f}s...", end="", flush=True)

        rate.sleep()

    print(" done")

    return {
        "timestamps": np.array(timestamps),
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "target": hold_pos,
    }


def analyze(data, rate_hz):
    """Compute and print jitter statistics."""
    positions = data["positions"]
    velocities = data["velocities"]
    target = data["target"]
    n_samples = len(positions)
    dt = 1.0 / rate_hz

    # --- Position jitter ---
    pos_mean = np.mean(positions)
    pos_std = np.std(positions)
    pos_ptp = np.ptp(positions)
    pos_error = positions - target

    print(f"\n{'='*60}")
    print(f"  POSITION JITTER ({n_samples} samples, {n_samples * dt:.1f}s)")
    print(f"{'='*60}")
    print(f"  Target position:  {target:>+10.4f} rad ({np.degrees(target):>+8.2f}°)")
    print(f"  Mean measured:    {pos_mean:>+10.4f} rad ({np.degrees(pos_mean):>+8.2f}°)")
    print(f"  Std deviation:    {pos_std:>10.4f} rad ({np.degrees(pos_std):>8.3f}°)")
    print(f"  Peak-to-peak:     {pos_ptp:>10.4f} rad ({np.degrees(pos_ptp):>8.3f}°)")
    print(f"  Mean error:       {np.mean(pos_error):>+10.4f} rad ({np.degrees(np.mean(pos_error)):>+8.3f}°)")
    print(f"  Max |error|:      {np.max(np.abs(pos_error)):>10.4f} rad ({np.degrees(np.max(np.abs(pos_error))):>8.3f}°)")

    # --- Velocity noise ---
    vel_mean = np.mean(velocities)
    vel_std = np.std(velocities)
    vel_rms = np.sqrt(np.mean(velocities**2))
    vel_max = np.max(np.abs(velocities))

    print(f"\n{'='*60}")
    print(f"  VELOCITY NOISE")
    print(f"{'='*60}")
    print(f"  Mean:     {vel_mean:>+10.4f} rad/s")
    print(f"  Std:      {vel_std:>10.4f} rad/s")
    print(f"  RMS:      {vel_rms:>10.4f} rad/s")
    print(f"  Max |v|:  {vel_max:>10.4f} rad/s")

    # --- Velocity jerk (smoothness) ---
    vel_diff = np.diff(velocities) / dt
    jerk_rms = np.sqrt(np.mean(vel_diff**2))

    print(f"\n{'='*60}")
    print(f"  VELOCITY JERK (lower = smoother)")
    print(f"{'='*60}")
    print(f"  RMS:  {jerk_rms:>10.2f} rad/s²")

    # --- FFT: dominant noise frequency ---
    print(f"\n{'='*60}")
    print(f"  DOMINANT NOISE FREQUENCY (FFT)")
    print(f"{'='*60}")

    freqs = np.fft.rfftfreq(n_samples, d=dt)
    signal = positions - pos_mean
    fft_mag = np.abs(np.fft.rfft(signal))
    fft_mag[0] = 0  # skip DC

    # Top 3 peaks
    top_indices = np.argsort(fft_mag)[::-1][:3]
    for rank, idx in enumerate(top_indices):
        amp_deg = np.degrees(fft_mag[idx] * 2 / n_samples)
        print(f"  #{rank+1}  {freqs[idx]:>8.2f} Hz  amplitude {amp_deg:.4f}°")

    # --- Timing analysis ---
    actual_dts = np.diff(data["timestamps"])
    print(f"\n{'='*60}")
    print(f"  TIMING")
    print(f"{'='*60}")
    print(f"  Target dt:   {dt*1000:>8.2f} ms ({rate_hz:.0f} Hz)")
    print(f"  Actual mean:  {np.mean(actual_dts)*1000:>8.2f} ms ({1/np.mean(actual_dts):.1f} Hz)")
    print(f"  Actual std:   {np.std(actual_dts)*1000:>8.2f} ms")
    print(f"  Max dt:       {np.max(actual_dts)*1000:>8.2f} ms")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Position noise: {np.degrees(pos_std):.3f}° std, {np.degrees(pos_ptp):.3f}° peak-to-peak")
    print(f"  Velocity noise: {vel_rms:.4f} rad/s RMS")
    print(f"  Dominant freq:  {freqs[top_indices[0]]:.2f} Hz")
    print(f"{'='*60}")


def save_results(data, path, args):
    """Save raw recording data to JSON."""
    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "channel": args.channel,
            "device_id": args.id,
            "kp": args.kp,
            "kd": args.kd,
            "torque_limit": args.torque_limit,
            "duration": args.duration,
            "rate_hz": args.rate,
            "n_samples": len(data["timestamps"]),
        },
        "target_position": float(data["target"]),
        "timestamps": data["timestamps"].tolist(),
        "positions": data["positions"].tolist(),
        "velocities": data["velocities"].tolist(),
    }

    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Saved raw data to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure single-joint jitter while holding position (raw CAN)")
    parser.add_argument("-c", "--channel", type=str, default="can2",
                        help="CAN channel (default: can2)")
    parser.add_argument("-i", "--id", type=int, default=1,
                        help="CAN device ID (default: 1)")
    parser.add_argument("--kp", type=float, default=DEFAULT_KP,
                        help=f"Position Kp (default: {DEFAULT_KP})")
    parser.add_argument("--kd", type=float, default=DEFAULT_KD,
                        help=f"Position Kd (default: {DEFAULT_KD})")
    parser.add_argument("--torque-limit", type=float, default=DEFAULT_TORQUE_LIMIT,
                        help=f"Torque limit in Nm (default: {DEFAULT_TORQUE_LIMIT})")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help=f"Recording duration in seconds (default: {DEFAULT_DURATION})")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE,
                        help=f"Sampling rate in Hz (default: {DEFAULT_RATE})")
    parser.add_argument("--save", type=str, default=None,
                        help="Save raw data to JSON file")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Single-Joint Jitter Measurement")
    print(f"  Channel: {args.channel}  ID: {args.id}")
    print(f"  Kp: {args.kp}  Kd: {args.kd}  Torque: {args.torque_limit} Nm")
    print(f"  Duration: {args.duration}s  Rate: {args.rate} Hz")
    print(f"{'='*60}\n")

    bus = recoil.Bus(channel=args.channel, bitrate=1000000)

    hold_pos = setup_actuator(bus, args.id, args.kp, args.kd, args.torque_limit)
    if hold_pos is None:
        bus.stop()
        return

    try:
        data = record_holding(bus, args.id, hold_pos, args.duration, args.rate)
        analyze(data, args.rate)

        if args.save:
            save_results(data, args.save, args)

    except KeyboardInterrupt:
        print("\n\n  Interrupted by Ctrl+C")

    finally:
        print("\n  Returning to IDLE mode...")
        bus.set_mode(args.id, recoil.Mode.IDLE)
        bus.stop()
        print("  Done.")


if __name__ == "__main__":
    main()
