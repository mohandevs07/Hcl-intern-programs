from pathlib import Path
import urllib.request
import cv2
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR
    / "dataset"
    / "super_resolution"
    / "low_resolution.jpg"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "ESPCN_x4.pb"

OUTPUT_PATH = BASE_DIR / "super_resolution_output.png"

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Input image not found: {IMAGE_PATH}"
    )

# Read low-resolution image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise ValueError("Could not read the input image.")

# Download ESPCN pretrained model if missing
MODEL_URL = (
    "https://raw.githubusercontent.com/"
    "fannymonori/TF-ESPCN/master/"
    "export/ESPCN_x4.pb"
)

if not MODEL_PATH.exists():
    print("Downloading ESPCN model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        str(MODEL_PATH)
    )

    print("Model downloaded successfully.")

# Check OpenCV super-resolution support
if not hasattr(cv2, "dnn_superres"):
    raise ImportError(
        "OpenCV super-resolution module is missing. "
        "Install opencv-contrib-python."
    )

# Create super-resolution object
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# Load pretrained model
sr.readModel(str(MODEL_PATH))

# Set ESPCN model and scale factor
sr.setModel("espcn", 4)

# Perform super-resolution
result = sr.upsample(image)

# Save output
cv2.imwrite(str(OUTPUT_PATH), result)

print("\nSuper-resolution completed!")
print("Original dimensions:", image.shape[1], "x", image.shape[0])
print("Output dimensions:", result.shape[1], "x", result.shape[0])
print("Saved output:", OUTPUT_PATH)

# Display comparison
original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Low-Resolution Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result_rgb)
plt.title("ESPCN Super-Resolution (4x)")
plt.axis("off")

plt.tight_layout()
plt.show()