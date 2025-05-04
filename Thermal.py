import time
import busio
import board
import adafruit_amg88xx
import numpy as np
import cv2

# Initialize I2C and AMG8833
i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_amg88xx.AMG88XX(i2c)

# Interpolation factor for upscaling 8x8 to 64x64
interp_factor = 8  # Upscale to 64x64

# Scale factor to enlarge the display feed
scale_factor = 8  # Scale the feed to 2x the original size

while True:
    # Capture AMG8833 thermal data (8x8 matrix)
    data = np.array(sensor.pixels)

    # Upscale to 64x64 using interpolation for better display
    data_interp = cv2.resize(data, (64, 64), interpolation=cv2.INTER_LINEAR)

    # Normalize the thermal data to 0-255 range
    thermal_image = cv2.normalize(data_interp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply the JET colormap to the thermal data
    thermal_image_colored = cv2.applyColorMap(thermal_image, cv2.COLORMAP_JET)

    # Resize the image to make it larger for display (optional)
    large_thermal_image = cv2.resize(thermal_image_colored, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Show the thermal image with JET colormap
    cv2.imshow("Thermal Feed", large_thermal_image)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)  # Add sleep for smoother display

cv2.destroyAllWindows()
