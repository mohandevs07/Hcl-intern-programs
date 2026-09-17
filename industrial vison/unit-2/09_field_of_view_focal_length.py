import math
import matplotlib.pyplot as plt

sensor_width = 36

focal_lengths = [18, 24, 35, 50, 85]

fov_values = []

for focal_length in focal_lengths:
    fov = 2 * math.degrees(
        math.atan(sensor_width / (2 * focal_length))
    )
    fov_values.append(fov)

for focal_length, fov in zip(focal_lengths, fov_values):
    print(
        f"Focal Length: {focal_length} mm "
        f"-> FOV: {fov:.2f} degrees"
    )

plt.plot(focal_lengths, fov_values, marker="o")

plt.xlabel("Focal Length (mm)")
plt.ylabel("Field of View (degrees)")
plt.title("FOV vs Focal Length")

plt.grid()
plt.show()