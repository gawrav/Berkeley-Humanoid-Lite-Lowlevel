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
import struct
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

    # Start in damping mode, read current position via SDO (no side effects)
    bus.set_mode(device_id, recoil.Mode.DAMPING)
    bus.feed(device_id)
    time.sleep(0.1)

    pos = bus.read_position_measured(device_id)
    if pos is None:
        print("  ERROR: Cannot read initial position")
        return None

    print(f"  Current position: {pos:.4f} rad ({np.degrees(pos):.2f}°)")
    return pos


def record_holding(bus, device_id, hold_pos, duration, rate_hz, include_torque=False, feed_only=False, pdo1_only=False, encoder_only=False):
    """Switch to position mode, hold current position, record noise.

    If feed_only=True, position_target is written once at the start, then
    the loop only sends HEARTBEAT (feed) to keep the watchdog alive and
    reads position/velocity via SDO parameter reads (no PDO_2 writes).
    This isolates whether repeated position_target writes affect jitter.

    If pdo1_only=True, sends PDO_1 echo frames instead of PDO_2. PDO_1
    goes through the full CAN handler path (TX response, watchdog reset)
    but does NOT write position_target or any controller field. Reads
    position/velocity via SDO. Isolates CAN handler overhead from the
    position_target write.

    If encoder_only=True, stays in DAMPING mode (no PD loop active).
    Reads position/velocity via SDO only. Measures pure encoder noise
    with no motor torque applied.
    """
    if not encoder_only:
        # Pre-load position_target via SDO BEFORE entering POSITION mode.
        # This avoids the race where the PD loop reads a stale position_target
        # during the gap between set_mode(POSITION) and the first PDO_2 write.
        bus.write_position_target(device_id, hold_pos)
        time.sleep(0.01)

        # Verify the write took effect
        readback = bus.read_position_target(device_id)
        if readback is not None:
            if abs(readback - hold_pos) > 0.01:
                print(f"  WARNING: position_target readback {readback:.4f} != hold_pos {hold_pos:.4f}")
            else:
                print(f"  Pre-loaded position_target: {readback:.4f} rad ({np.degrees(readback):.2f}°) [OK]")
        else:
            print(f"  WARNING: position_target readback failed")

        # Now switch to POSITION mode. PositionController_reset() does NOT
        # touch position_target, so our pre-loaded value is preserved.
        bus.set_mode(device_id, recoil.Mode.POSITION)
        time.sleep(0.01)

        # Send one PDO_2 to confirm the target and get initial position/velocity
        bus.write_read_pdo_2(device_id, hold_pos, 0.0)
        bus.feed(device_id)
        time.sleep(0.05)
    # else: stay in DAMPING mode (set by setup_actuator), no PD

    rate = RateLimiter(frequency=rate_hz)
    steps = int(duration * rate_hz)

    timestamps = []
    positions = []
    velocities = []
    torques = []
    position_targets = []

    if encoder_only:
        mode_label = "ENCODER-ONLY"
    elif pdo1_only:
        mode_label = "PDO_1-ONLY"
    elif feed_only:
        mode_label = "FEED-ONLY"
    else:
        mode_label = "PDO_2"
    print(f"  Recording {duration:.1f}s at {rate_hz:.0f} Hz ({steps} samples) [{mode_label}]...")

    t0 = time.perf_counter()

    for i in range(steps):
        bus.feed(device_id)

        if encoder_only:
            # Pure encoder read — no PD, no targets, no position mode
            pos = bus.read_position_measured(device_id)
            vel = bus.read_velocity_measured(device_id)
        elif pdo1_only:
            # Send PDO_1 echo frame (full CAN handler path, TX response,
            # watchdog reset, but NO position_target write)
            bus.transmit(recoil.CANFrame(
                device_id, recoil.Function.RECEIVE_PDO_1,
                size=8, data=struct.pack("<ff", hold_pos, 0.0)))
            bus.receive(filter_device_id=device_id, timeout=0.001)
            # Read position/velocity via SDO (PDO_1 response is just echo)
            pos = bus.read_position_measured(device_id)
            vel = bus.read_velocity_measured(device_id)
        elif feed_only:
            # Read position/velocity via SDO parameter reads (no target write)
            pos = bus.read_position_measured(device_id)
            vel = bus.read_velocity_measured(device_id)
        else:
            # Normal: write target + read measured via PDO_2
            pos, vel = bus.write_read_pdo_2(device_id, hold_pos, 0.0)

        # Read measured torque from motor controller (optional extra CAN read)
        torque = bus.read_torque_measured(device_id) if include_torque else None
        # Read position_target from firmware to verify what the PD loop sees
        pos_target = bus.read_position_target(device_id)

        timestamps.append(time.perf_counter() - t0)
        if pos is not None:
            positions.append(pos)
        else:
            positions.append(positions[-1] if positions else hold_pos)
        if vel is not None:
            velocities.append(vel)
        else:
            velocities.append(velocities[-1] if velocities else 0.0)
        torques.append(torque if torque is not None else 0.0)
        position_targets.append(pos_target if pos_target is not None else hold_pos)

        if (i + 1) % int(rate_hz) == 0:
            print(f"    {(i + 1) / rate_hz:.0f}s...", end="", flush=True)

        rate.sleep()

    print(" done")

    return {
        "timestamps": np.array(timestamps),
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "torques": np.array(torques),
        "position_targets": np.array(position_targets),
        "target": hold_pos,
    }


def analyze(data, rate_hz):
    """Compute and print jitter statistics."""
    positions = data["positions"]
    velocities = data["velocities"]
    torques = data["torques"]
    position_targets = data["position_targets"]
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

    # --- Firmware position_target ---
    pt_mean = np.mean(position_targets)
    pt_std = np.std(position_targets)
    pt_ptp = np.ptp(position_targets)
    print(f"\n{'='*60}")
    print(f"  FIRMWARE position_target (SDO readback)")
    print(f"{'='*60}")
    print(f"  Script hold_pos:  {target:>+10.4f} rad ({np.degrees(target):>+8.2f}°)")
    print(f"  FW mean:          {pt_mean:>+10.4f} rad ({np.degrees(pt_mean):>+8.2f}°)")
    print(f"  FW std:           {pt_std:>10.4f} rad ({np.degrees(pt_std):>8.3f}°)")
    print(f"  FW peak-to-peak:  {pt_ptp:>10.4f} rad ({np.degrees(pt_ptp):>8.3f}°)")
    print(f"  FW first:         {position_targets[0]:>+10.4f} rad ({np.degrees(position_targets[0]):>+8.2f}°)")
    print(f"  FW last:          {position_targets[-1]:>+10.4f} rad ({np.degrees(position_targets[-1]):>+8.2f}°)")

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

    # --- Torque ---
    has_torque = np.any(torques != 0)
    if has_torque:
        torque_mean = np.mean(torques)
        torque_std = np.std(torques)
        torque_rms = np.sqrt(np.mean(torques**2))
        torque_max = np.max(np.abs(torques))
        torque_ptp = np.ptp(torques)

        print(f"\n{'='*60}")
        print(f"  TORQUE (measured from motor)")
        print(f"{'='*60}")
        print(f"  Mean:         {torque_mean:>+10.4f} Nm")
        print(f"  Std:          {torque_std:>10.4f} Nm")
        print(f"  RMS:          {torque_rms:>10.4f} Nm")
        print(f"  Max |torque|: {torque_max:>10.4f} Nm")
        print(f"  Peak-to-peak: {torque_ptp:>10.4f} Nm")

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
    if has_torque:
        print(f"  Torque applied: {torque_mean:+.4f} Nm mean, {torque_rms:.4f} Nm RMS, {torque_ptp:.4f} Nm peak-to-peak")
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

    if np.any(data["torques"] != 0):
        result["torques"] = data["torques"].tolist()

    result["position_targets"] = data["position_targets"].tolist()

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
    parser.add_argument("--include-torque", action="store_true",
                        help="Read measured torque each cycle (extra CAN transaction)")
    parser.add_argument("--feed-only", action="store_true",
                        help="Write position once, then only send feed() keepalives (no PDO_2 writes)")
    parser.add_argument("--pdo1-only", action="store_true",
                        help="Send PDO_1 echo frames (full CAN handler path, but no position_target write)")
    parser.add_argument("--encoder-only", action="store_true",
                        help="Stay in DAMPING mode (no PD), read position via SDO only (pure encoder noise)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Single-Joint Jitter Measurement")
    print(f"  Channel: {args.channel}  ID: {args.id}")
    print(f"  Kp: {args.kp}  Kd: {args.kd}  Torque: {args.torque_limit} Nm")
    print(f"  Duration: {args.duration}s  Rate: {args.rate} Hz")
    if args.encoder_only:
        print(f"  Mode: ENCODER-ONLY (DAMPING mode, no PD, pure encoder noise)")
    elif args.pdo1_only:
        print(f"  Mode: PDO_1-ONLY (echo frames, no position_target write)")
    elif args.feed_only:
        print(f"  Mode: FEED-ONLY (position written once, then heartbeat only)")
    print(f"{'='*60}\n")

    bus = recoil.Bus(channel=args.channel, bitrate=1000000)

    hold_pos = setup_actuator(bus, args.id, args.kp, args.kd, args.torque_limit)
    if hold_pos is None:
        bus.stop()
        return

    try:
        data = record_holding(bus, args.id, hold_pos, args.duration, args.rate, args.include_torque, args.feed_only, args.pdo1_only, args.encoder_only)
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
