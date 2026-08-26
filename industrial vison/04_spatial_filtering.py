import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/industrial.jpg")

if image is None:
    raise FileNotFoundError("Image not found.")

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# Mean filtering
mean_filtered = cv2.blur(
    image_rgb,
    (5, 5)
)

# Gaussian filtering
gaussian_filtered = cv2.GaussianBlur(
    image_rgb,
    (5, 5),
    0
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mean_filtered)
plt.title("Mean Filter")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gaussian_filtered)
plt.title("Gaussian Filter")
plt.axis("off")

plt.tight_layout()
plt.show()