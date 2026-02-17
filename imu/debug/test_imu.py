import serial
ser = serial.Serial('/dev/ttyACM0', 1000000)
print(ser.read(28).hex())
