import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/industrial.jpg")

if image is None:
    raise FileNotFoundError("Image not found.")

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# -------------------------
# Sobel
# -------------------------

sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

sobel_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

sobel = cv2.addWeighted(
    sobel_x,
    0.5,
    sobel_y,
    0.5,
    0
)

# -------------------------
# Canny
# -------------------------

canny = cv2.Canny(
    gray,
    50,
    150
)

# -------------------------
# Laplacian
# -------------------------

laplacian = cv2.Laplacian(
    gray,
    cv2.CV_64F
)

laplacian = cv2.convertScaleAbs(
    laplacian
)

# -------------------------
# Display
# -------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(canny, cmap="gray")
plt.title("Canny")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian")
plt.axis("off")

plt.tight_layout()
plt.show()