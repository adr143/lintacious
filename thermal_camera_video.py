import time
import busio
import board
import adafruit_amg88xx
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Initialize I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize AMG8833 sensor
sensor = adafruit_amg88xx.AMG88XX(i2c)

# Set up the plot for the heatmap display
plt.ion()  # Interactive mode ON
fig, ax = plt.subplots()

# Interpolation factor for upscaling the 8x8 matrix to a larger image
interp = 4  # Upscaling to 32x32
im = ax.imshow(np.zeros((8 * interp, 8 * interp)), cmap="inferno", vmin=20, vmax=40)

# Set axis labels
ax.set_title('Thermal Heatmap')
ax.set_xlabel('Pixel X')
ax.set_ylabel('Pixel Y')

# Create a scatter plot for the coldest point marker
cold_point_marker, = ax.plot([], [], 'rx', markersize=10)  # Red cross for coldest point

while True:
    # Capture the thermal data from the AMG8833 sensor
    data = np.array(sensor.pixels)

    # Interpolate the 8x8 thermal data to a higher resolution (e.g., 32x32)
    data_interp = zoom(data, interp)

    # Update the image with the new thermal data
    im.set_data(data_interp)

    # Optional: Adjust the color scale dynamically based on the data (e.g., if you want to track min/max temperature in real-time)
    im.set_clim(np.min(data_interp), np.max(data_interp))

    # Find the coldest point (minimum temperature) in the data
    coldest_value = np.min(data)  # The coldest temperature
    coldest_index = np.unravel_index(np.argmin(data), data.shape)  # Index of coldest point in the 8x8 array

    # Map the coldest index to the interpolated image coordinates
    coldest_x = coldest_index[1] * interp + interp / 2  # Calculate X position in the larger image
    coldest_y = coldest_index[0] * interp + interp / 2  # Calculate Y position in the larger image

    # Update the coldest point marker position (passing sequences)
    cold_point_marker.set_data([coldest_x], [coldest_y])  # Use lists to pass sequences

    # Pause briefly to update the display
    plt.pause(0.1)  # You can adjust the pause time for smoother visualization
