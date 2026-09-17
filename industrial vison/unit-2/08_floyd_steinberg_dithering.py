import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/industrial.jpg", cv2.IMREAD_GRAYSCALE)

image = image.astype(float)

height, width = image.shape

for y in range(height):
    for x in range(width):

        old_pixel = image[y, x]

        if old_pixel < 128:
            new_pixel = 0
        else:
            new_pixel = 255

        image[y, x] = new_pixel

        error = old_pixel - new_pixel

        if x + 1 < width:
            image[y, x + 1] += error * 7 / 16

        if y + 1 < height and x - 1 >= 0:
            image[y + 1, x - 1] += error * 3 / 16

        if y + 1 < height:
            image[y + 1, x] += error * 5 / 16

        if y + 1 < height and x + 1 < width:
            image[y + 1, x + 1] += error * 1 / 16

image = np.clip(image, 0, 255).astype(np.uint8)

original = cv2.imread(
    "../images/industrial.jpg",
    cv2.IMREAD_GRAYSCALE
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image, cmap="gray")
plt.title("Floyd-Steinberg Dithered")
plt.axis("off")

plt.tight_layout()
plt.show()