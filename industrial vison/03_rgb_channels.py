import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/industrial.jpg")

if image is None:
    raise FileNotFoundError("Image not found.")

# OpenCV stores images as BGR
blue = image[:, :, 0]
green = image[:, :, 1]
red = image[:, :, 2]

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(red, cmap="gray")
plt.title("Red Channel")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(green, cmap="gray")
plt.title("Green Channel")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(blue, cmap="gray")
plt.title("Blue Channel")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Image shape:", image.shape)
print("Red channel shape:", red.shape)
print("Green channel shape:", green.shape)
print("Blue channel shape:", blue.shape)