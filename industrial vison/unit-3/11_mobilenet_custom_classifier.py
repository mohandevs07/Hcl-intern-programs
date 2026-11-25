from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Project paths
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "classification" / "train"
VAL_DIR = BASE_DIR / "dataset" / "classification" / "validation"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 5

# Check dataset folders
if not TRAIN_DIR.exists() or not VAL_DIR.exists():
    raise FileNotFoundError("Classification dataset folders were not found.")

# Load training images
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Load validation images
val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

class_names = train_data.class_names
print("Classes:", class_names)

# Preprocess images for MobileNetV2
train_data = train_data.map(
    lambda x, y: (preprocess_input(x), y)
)

val_data = val_data.map(
    lambda x, y: (preprocess_input(x), y)
)

# Load pretrained MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# Build classifier
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save model
save_path = MODEL_DIR / "mobilenet_classifier.keras"
model.save(save_path)

print("\nTraining completed!")
print("Model saved at:", save_path)