# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Actuator Delay Measurement Script

Measures the end-to-end delay between sending a position command and seeing the
motor respond. Uses two methods:
  1. Step response: sends a sudden position step and measures time until motion starts
  2. Cross-correlation: sends a PRBS signal and correlates command vs response

Results help calibrate the DelayedPDActuator in Isaac Lab simulation (currently 0-80ms).

Usage:
    uv run scripts/motor/measure_delay.py -c can1 -i 1
    uv run scripts/motor/measure_delay.py -c can1 -i 1 --method cross-correlation
    uv run scripts/motor/measure_delay.py -c can1 -i 1 --method both

Joint ID Reference:
    can1: left_hip_roll(1), left_hip_yaw(3), left_hip_pitch(5),
          left_knee_pitch(7), left_ankle_pitch(11), left_ankle_roll(13)
    can2: right_hip_roll(2), right_hip_yaw(4), right_hip_pitch(6),
          right_knee_pitch(8), right_ankle_pitch(12), right_ankle_roll(14)
"""

import argparse
import time
import numpy as np

from loop_rate_limiters import RateLimiter
import berkeley_humanoid_lite_lowlevel.recoil as recoil


# Safe operating parameters
KP = 10.0
KD = 2.0
TORQUE_LIMIT = 3.0
CONTROL_RATE = 200.0  # Hz (5ms per step, matching move_actuator.py)

# Step response parameters
STEP_AMPLITUDE = 0.15  # rad — small enough to be safe, large enough to measure
STEP_SETTLE_TIME = 2.0  # seconds to wait before each step
STEP_RECORD_TIME = 0.5  # seconds to record after step
NUM_STEPS = 20  # number of step trials
VELOCITY_THRESHOLD = 0.05  # rad/s — minimum velocity to count as "motor started moving"

# Cross-correlation parameters
PRBS_AMPLITUDE = 0.1  # rad — peak-to-peak of the random signal
PRBS_DURATION = 10.0  # seconds of PRBS recording
PRBS_SWITCH_RATE = 10.0  # Hz — how fast the PRBS signal switches


def setup_actuator(bus, device_id):
    """Configure actuator for delay measurement."""
    if not bus.ping(device_id):
        print(f"ERROR: Actuator {device_id} not responding")
        return False

    print(f"Actuator {device_id} online. Configuring: kp={KP}, kd={KD}, torque_limit={TORQUE_LIMIT}")
    bus.write_position_kp(device_id, KP)
    time.sleep(0.001)
    bus.write_position_kd(device_id, KD)
    time.sleep(0.001)
    bus.write_torque_limit(device_id, TORQUE_LIMIT)
    time.sleep(0.001)

    # Start in damping mode, read current position
    bus.set_mode(device_id, recoil.Mode.DAMPING)
    bus.feed(device_id)
    time.sleep(0.1)

    pos, vel = bus.write_read_pdo_2(device_id, 0, 0)
    if pos is None:
        print("ERROR: Cannot read initial position")
        return False

    print(f"Initial position: {pos:.4f} rad")
    return True


def measure_step_response(bus, device_id):
    """Measure delay using step responses.

    Procedure:
      1. Hold at position A for STEP_SETTLE_TIME
      2. Send sudden step to position B
      3. Record timestamps + measured position/velocity at CONTROL_RATE
      4. Find when velocity exceeds threshold → that's the delay
      5. Repeat NUM_STEPS times, alternating direction
    """
    print(f"\n{'='*60}")
    print("Step Response Delay Measurement")
    print(f"{'='*60}")
    print(f"  Amplitude: ±{STEP_AMPLITUDE} rad")
    print(f"  Trials: {NUM_STEPS}")
    print(f"  Control rate: {CONTROL_RATE} Hz")
    print(f"  Velocity threshold: {VELOCITY_THRESHOLD} rad/s")

    # Enable position control
    pos, vel = bus.write_read_pdo_2(device_id, 0, 0)
    center = pos if pos is not None else 0.0
    bus.set_mode(device_id, recoil.Mode.POSITION)
    bus.feed(device_id)

    rate = RateLimiter(frequency=CONTROL_RATE)
    delays_ms = []

    for trial in range(NUM_STEPS):
        direction = 1 if trial % 2 == 0 else -1
        pos_before = center - direction * STEP_AMPLITUDE / 2
        pos_after = center + direction * STEP_AMPLITUDE / 2

        # Settle at starting position
        print(f"\n  Trial {trial+1}/{NUM_STEPS}: settling at {pos_before:.3f} rad...", end="", flush=True)
        settle_steps = int(STEP_SETTLE_TIME * CONTROL_RATE)
        for _ in range(settle_steps):
            bus.write_read_pdo_2(device_id, pos_before, 0.0)
            bus.feed(device_id)
            rate.sleep()

        # Send step and record
        record_steps = int(STEP_RECORD_TIME * CONTROL_RATE)
        timestamps = []
        positions = []
        velocities = []

        step_time = time.perf_counter()
        for i in range(record_steps):
            bus.feed(device_id)
            measured_pos, measured_vel = bus.write_read_pdo_2(device_id, pos_after, 0.0)
            t = time.perf_counter()

            if measured_pos is not None and measured_vel is not None:
                timestamps.append(t - step_time)
                positions.append(measured_pos)
                velocities.append(measured_vel)

            rate.sleep()

        # Find delay: first time velocity exceeds threshold in the step direction
        delay_found = False
        for j, (t, v) in enumerate(zip(timestamps, velocities)):
            if direction * v > VELOCITY_THRESHOLD:
                delay_ms = t * 1000
                delays_ms.append(delay_ms)
                print(f" delay = {delay_ms:.1f} ms")
                delay_found = True
                break

        if not delay_found:
            print(f" no response detected (max vel: {max(abs(v) for v in velocities):.3f} rad/s)")

    # Return to center
    for _ in range(int(0.5 * CONTROL_RATE)):
        bus.write_read_pdo_2(device_id, center, 0.0)
        bus.feed(device_id)
        rate.sleep()

    # Report results
    if delays_ms:
        delays = np.array(delays_ms)
        print(f"\n{'='*60}")
        print("Step Response Results")
        print(f"{'='*60}")
        print(f"  Successful trials: {len(delays)}/{NUM_STEPS}")
        print(f"  Mean delay:   {np.mean(delays):.1f} ms")
        print(f"  Std delay:    {np.std(delays):.1f} ms")
        print(f"  Min delay:    {np.min(delays):.1f} ms")
        print(f"  Max delay:    {np.max(delays):.1f} ms")
        print(f"  Median delay: {np.median(delays):.1f} ms")
        print(f"\n  Recommended sim range: 0 to {np.max(delays) * 1.5:.0f} ms")
        print(f"  (= 0 to {int(np.max(delays) * 1.5 / 5)} physics steps at 5ms sim_dt)")
    else:
        print("\n  No delays measured — motor may not be responding")

    return delays_ms


def measure_cross_correlation(bus, device_id):
    """Measure delay using cross-correlation with a PRBS signal.

    Procedure:
      1. Generate a pseudo-random binary sequence (PRBS) of position commands
      2. Send commands at CONTROL_RATE while recording measured position
      3. Compute cross-correlation between command and response
      4. Peak lag = system delay
    """
    print(f"\n{'='*60}")
    print("Cross-Correlation Delay Measurement")
    print(f"{'='*60}")
    print(f"  PRBS amplitude: ±{PRBS_AMPLITUDE} rad")
    print(f"  Duration: {PRBS_DURATION} s")
    print(f"  Control rate: {CONTROL_RATE} Hz")

    # Read current position as center
    pos, vel = bus.write_read_pdo_2(device_id, 0, 0)
    center = pos if pos is not None else 0.0

    # Enable position control
    bus.set_mode(device_id, recoil.Mode.POSITION)
    bus.feed(device_id)

    # Settle at center
    rate = RateLimiter(frequency=CONTROL_RATE)
    for _ in range(int(1.0 * CONTROL_RATE)):
        bus.write_read_pdo_2(device_id, center, 0.0)
        bus.feed(device_id)
        rate.sleep()

    # Generate PRBS signal
    total_steps = int(PRBS_DURATION * CONTROL_RATE)
    switch_interval = int(CONTROL_RATE / PRBS_SWITCH_RATE)
    rng = np.random.default_rng(42)

    commands = np.zeros(total_steps)
    current_val = PRBS_AMPLITUDE
    for i in range(total_steps):
        if i % switch_interval == 0:
            current_val = rng.choice([-PRBS_AMPLITUDE, PRBS_AMPLITUDE])
        commands[i] = center + current_val

    # Send PRBS and record
    print("  Recording...", end="", flush=True)
    timestamps = []
    measured_positions = []

    for i in range(total_steps):
        bus.feed(device_id)
        measured_pos, measured_vel = bus.write_read_pdo_2(device_id, commands[i], 0.0)
        t = time.perf_counter()

        timestamps.append(t)
        if measured_pos is not None:
            measured_positions.append(measured_pos)
        else:
            measured_positions.append(measured_positions[-1] if measured_positions else center)

        rate.sleep()

        if (i + 1) % int(CONTROL_RATE) == 0:
            print(f" {(i+1)/CONTROL_RATE:.0f}s", end="", flush=True)

    print(" done")

    # Return to center
    for _ in range(int(0.5 * CONTROL_RATE)):
        bus.write_read_pdo_2(device_id, center, 0.0)
        bus.feed(device_id)
        rate.sleep()

    # Compute cross-correlation
    cmd_signal = np.array(commands) - center
    pos_signal = np.array(measured_positions) - center

    # Normalize
    cmd_signal = cmd_signal - np.mean(cmd_signal)
    pos_signal = pos_signal - np.mean(pos_signal)

    correlation = np.correlate(pos_signal, cmd_signal, mode='full')
    lags = np.arange(-len(cmd_signal) + 1, len(cmd_signal))

    # Only look at positive lags (response comes after command)
    positive_mask = lags >= 0
    positive_lags = lags[positive_mask]
    positive_corr = correlation[positive_mask]

    # Find peak in first 100ms worth of lags
    max_lag_samples = int(0.1 * CONTROL_RATE)  # 100ms
    search_corr = positive_corr[:max_lag_samples]
    peak_idx = np.argmax(search_corr)
    peak_lag_samples = positive_lags[peak_idx]
    delay_ms = peak_lag_samples / CONTROL_RATE * 1000

    print(f"\n{'='*60}")
    print("Cross-Correlation Results")
    print(f"{'='*60}")
    print(f"  Peak lag: {peak_lag_samples} samples = {delay_ms:.1f} ms")
    print(f"  Correlation at peak: {search_corr[peak_idx]:.4f}")
    print(f"  Sample rate: {CONTROL_RATE} Hz ({1000/CONTROL_RATE:.1f} ms/sample)")
    print(f"\n  Recommended sim range: 0 to {delay_ms * 2:.0f} ms")
    print(f"  (= 0 to {int(delay_ms * 2 / 5)} physics steps at 5ms sim_dt)")

    return delay_ms


def main():
    parser = argparse.ArgumentParser(description="Measure actuator delay for sim-to-real calibration")
    parser.add_argument("-c", "--channel", help="CAN transport channel", type=str, default="can0")
    parser.add_argument("-i", "--id", help="CAN device ID", type=int, default=1)
    parser.add_argument(
        "--method", type=str, default="both",
        choices=["step", "cross-correlation", "both"],
        help="Measurement method (default: both)"
    )
    args = parser.parse_args()

    bus = recoil.Bus(channel=args.channel, bitrate=1000000)
    device_id = args.id

    if not setup_actuator(bus, device_id):
        bus.stop()
        return

    try:
        if args.method in ["step", "both"]:
            measure_step_response(bus, device_id)

        if args.method in ["cross-correlation", "both"]:
            measure_cross_correlation(bus, device_id)

    except KeyboardInterrupt:
        print("\n\nInterrupted by Ctrl+C")

    finally:
        print("\nReturning to IDLE mode...")
        bus.set_mode(device_id, recoil.Mode.IDLE)
        bus.stop()
        print("Done.")


if __name__ == "__main__":
    main()
