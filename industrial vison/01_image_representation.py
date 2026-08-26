import cv2
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("images/industrial.jpg")

if image is None:
    raise FileNotFoundError("Image not found.")

# Convert BGR to RGB
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert grayscale to binary
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Display
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

# Properties
print("----- RGB IMAGE -----")
print("Shape:", rgb.shape)
print("Dimensions:", rgb.shape[:2])
print("Channels:", rgb.shape[2])
print("Pixel value:", rgb[100, 100])

print("\n----- GRAYSCALE IMAGE -----")
print("Shape:", gray.shape)
print("Dimensions:", gray.shape)
print("Channels: 1")
print("Pixel value:", gray[100, 100])

print("\n----- BINARY IMAGE -----")
print("Shape:", binary.shape)
print("Dimensions:", binary.shape)
print("Channels: 1")
print("Unique values:", set(binary.flatten()))