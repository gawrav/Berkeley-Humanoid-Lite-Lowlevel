# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
IMU Test Script

Displays all IMU values used by the locomotion policy:
- Quaternion (w, x, y, z) - raw and corrected
- Angular velocity (rad/s) - as used by model
- Euler angles (deg) - for human readability

Usage:
    uv run scripts/test_imu.py
"""

import time
import struct
import math
import numpy as np
import scipy.spatial.transform as st

import serial

SYNC_1 = b'\x75'
SYNC_2 = b'\x65'

# IMU roll offset correction (must match humanoid.py)
IMU_ROLL_OFFSET_DEG = 7.4


def quaternion_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_rotate_inverse(q, v):
    """Rotate a vector by the inverse of a quaternion (w, x, y, z format).
    This is used to compute projected gravity - same as rl_controller.py"""
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * (np.dot(q_vec, v)) * 2.0
    return a - b + c


# Gravity vector (what policy uses)
GRAVITY_VECTOR = np.array([0., 0., -1.], dtype=np.float32)


# Pre-compute correction quaternion
_roll_offset_rad = np.deg2rad(-IMU_ROLL_OFFSET_DEG)
CORRECTION_QUAT = np.array([
    np.cos(_roll_offset_rad / 2),
    np.sin(_roll_offset_rad / 2),
    0.0,
    0.0
])


ser = serial.Serial("/dev/ttyACM0", 1000000, timeout=0.001)

print("\n=== IMU Test ===")
print("Showing values as used by locomotion policy")
print("Press Ctrl+C to stop\n")

try:
    while True:
        t = time.perf_counter()

        sync_1 = ser.read(1)
        if not sync_1 or sync_1 != SYNC_1:
            continue

        sync_2 = ser.read(1)
        if not sync_2 or sync_2 != SYNC_2:
            continue

        size = ser.read(2)
        data_buffer = ser.read(28)

        data = struct.unpack("f"*7, data_buffer)

        w, x, y, z, rx, ry, rz = data
        raw_quat = np.array([w, x, y, z])

        # Apply correction (same as humanoid.py)
        corrected_quat = quaternion_multiply(CORRECTION_QUAT, raw_quat)

        # Angular velocity: IMU outputs deg/s, model expects rad/s
        rx_rad = math.radians(rx)
        ry_rad = math.radians(ry)
        rz_rad = math.radians(rz)

        # Euler angles for human readability
        raw_euler = st.Rotation.from_quat([w, x, y, z], scalar_first=True).as_euler("xyz", degrees=True)
        corrected_euler = st.Rotation.from_quat(corrected_quat, scalar_first=True).as_euler("xyz", degrees=True)

        # Projected gravity (what policy actually receives)
        projected_gravity = quat_rotate_inverse(corrected_quat, GRAVITY_VECTOR)

        freq = 1 / (time.perf_counter() - t)

        # Clear and print multi-line status
        print("\033[H\033[J", end="")  # Clear screen
        print("=" * 75)
        print(f"{'IMU STATUS':^75}")
        print("=" * 75)
        print()
        print(f"  RAW Quaternion:        [{w:+.4f}, {x:+.4f}, {y:+.4f}, {z:+.4f}]")
        print(f"  RAW Euler (deg):       roll={raw_euler[0]:+.2f}, pitch={raw_euler[1]:+.2f}, yaw={raw_euler[2]:+.2f}")
        print()
        print(f"  CORRECTED Quaternion:  [{corrected_quat[0]:+.4f}, {corrected_quat[1]:+.4f}, {corrected_quat[2]:+.4f}, {corrected_quat[3]:+.4f}]")
        print(f"  CORRECTED Euler (deg): roll={corrected_euler[0]:+.2f}, pitch={corrected_euler[1]:+.2f}, yaw={corrected_euler[2]:+.2f}")
        print(f"  (ideal when level:     roll=+0.00, pitch=+0.00)")
        print()
        print(f"  Angular Vel (rad/s):   [{rx_rad:+.4f}, {ry_rad:+.4f}, {rz_rad:+.4f}]")
        print()
        print(f"  PROJECTED GRAVITY:     [{projected_gravity[0]:+.4f}, {projected_gravity[1]:+.4f}, {projected_gravity[2]:+.4f}]")
        print(f"  (ideal when level:     [+0.0000, +0.0000, -1.0000])")
        print()
        print(f"  Correction applied:    {IMU_ROLL_OFFSET_DEG}° roll offset")
        print(f"  Update freq:           {freq:.0f} Hz")
        print()
        print("=" * 75)
        print("Press Ctrl+C to stop")

except KeyboardInterrupt:
    print("\n\nStopped.")

ser.close()
