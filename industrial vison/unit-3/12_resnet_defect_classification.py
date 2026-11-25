from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "classification" / "train"
VAL_DIR = BASE_DIR / "dataset" / "classification" / "validation"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

if not TRAIN_DIR.exists() or not VAL_DIR.exists():
    raise FileNotFoundError("Classification dataset folders were not found.")

# Load datasets
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("Classes:", train_data.class_names)

# ResNet preprocessing
train_data = train_data.map(
    lambda x, y: (preprocess_input(x), y)
)

val_data = val_data.map(
    lambda x, y: (preprocess_input(x), y)
)

# Load pretrained ResNet50
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Initially freeze all layers
base_model.trainable = False

# Build classification model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nInitial training...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=3
)

# Fine-tune the last 30 ResNet layers
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nFine-tuning ResNet50...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=3
)

# Save model
save_path = MODEL_DIR / "resnet_defect_classifier.keras"
model.save(save_path)

print("\nResNet fine-tuning completed!")
print("Model saved at:", save_path)