import cv2
import matplotlib.pyplot as plt

image = cv2.imread("../images/industrial.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower = (90, 50, 50)
upper = (130, 255, 255)

mask = cv2.inRange(hsv, lower, upper)

segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(hsv)
plt.title("HSV Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(mask, cmap="gray")
plt.title("Segmentation Mask")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(segmented)
plt.title("Segmented Object")
plt.axis("off")

plt.tight_layout()
plt.show()