from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "dataset" / "segmentation" / "images"
MASK_DIR = BASE_DIR / "dataset" / "segmentation" / "masks"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 128
EPOCHS = 10

# Load images and masks
images = []
masks = []

image_files = sorted(IMAGE_DIR.glob("*.png"))

if not image_files:
    raise FileNotFoundError("No PNG images found in segmentation/images.")

for image_path in image_files:
    mask_path = MASK_DIR / image_path.name

    if not mask_path.exists():
        print("Mask missing:", image_path.name)
        continue

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    mask = cv2.resize(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    # Normalize image and convert mask to binary
    image = image.astype("float32") / 255.0
    mask = (mask > 127).astype("float32")

    images.append(image)
    masks.append(mask)

X = np.array(images)
Y = np.array(masks).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

if len(X) < 2:
    raise ValueError("Not enough matching image and mask pairs.")

print("Total image-mask pairs:", len(X))

# Split dataset
X_train, X_val, Y_train, Y_val = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# U-Net model
def build_unet():

    inputs = layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3))

    # Encoder
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck
    b = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    b = layers.Conv2D(128, 3, activation="relu", padding="same")(b)

    # Decoder
    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(u1)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)

    u2 = layers.UpSampling2D()(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(u2)
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(c4)

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c4)

    return models.Model(inputs, outputs)

model = build_unet()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train U-Net
model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=4
)

# Save model
save_path = MODEL_DIR / "unet_segmentation.keras"
model.save(save_path)

# Generate a sample predicted mask
prediction = model.predict(X_val[:1])

predicted_mask = (prediction[0, :, :, 0] > 0.5).astype("uint8") * 255

output_path = BASE_DIR / "unet_predicted_mask.png"
cv2.imwrite(str(output_path), predicted_mask)

print("\nU-Net training completed!")
print("Model saved at:", save_path)
print("Predicted mask saved at:", output_path)