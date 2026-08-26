import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/industrial.jpg")

if image is None:
    raise FileNotFoundError("Image not found.")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(rgb)
plt.title("RGB")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(binary, cmap="gray")
plt.title("Binary")
plt.axis("off")

plt.tight_layout()
plt.show()

print("RGB shape:", rgb.shape)
print("Grayscale shape:", gray.shape)
print("Binary shape:", binary.shape)

print("\nNumPy memory representation:")
print("RGB:", rgb.nbytes, "bytes")
print("Grayscale:", gray.nbytes, "bytes")
print("Binary:", binary.nbytes, "bytes")
