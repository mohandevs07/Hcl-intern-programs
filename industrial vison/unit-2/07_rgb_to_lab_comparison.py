import cv2
import matplotlib.pyplot as plt

image = cv2.imread("../images/industrial.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

L, A, B = cv2.split(lab)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("RGB Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(lab)
plt.title("LAB Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(L, cmap="gray")
plt.title("L - Lightness")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(A, cmap="gray")
plt.title("A Channel")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(B, cmap="gray")
plt.title("B Channel")
plt.axis("off")

plt.tight_layout()
plt.show()