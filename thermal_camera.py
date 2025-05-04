import time
import busio
import board
import adafruit_amg88xx
import numpy as np

i2c = busio.I2C(board.SCL, board.SDA)

sensor = adafruit_amg88xx.AMG88XX(i2c)

while True:
        thermal_data = np. array(sensor.pixels)
        print(thermal_data)
        time.sleep()