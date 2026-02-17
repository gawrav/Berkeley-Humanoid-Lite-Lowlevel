#!/usr/bin/env python3
import serial
import struct

ser = serial.Serial('/dev/ttyACM0', 1000000)

while True:
    # Find sync bytes
    if ser.read(1) == b'\x75':
        if ser.read(1) == b'\x65':
            # Read size (2 bytes) + data (28 bytes)
            data = ser.read(30)
            size = struct.unpack('<H', data[0:2])[0]
            
            # Unpack floats
            rw, rx, ry, rz = struct.unpack('<ffff', data[2:18])
            vx, vy, vz = struct.unpack('<fff', data[18:30])
            
            print(f"Quat: w={rw:.3f} x={rx:.3f} y={ry:.3f} z={rz:.3f} | Gyro: x={vx:.3f} y={vy:.3f} z={vz:.3f}")

