from pathlib import Path
import yaml
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

YOLO_DIR = BASE_DIR / "dataset" / "yolo"
TRAIN_IMAGES = YOLO_DIR / "images" / "train"
VAL_IMAGES = YOLO_DIR / "images" / "val"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Check dataset
if not TRAIN_IMAGES.exists() or not VAL_IMAGES.exists():
    raise FileNotFoundError("YOLO image folders were not found.")

# Create a YAML file with absolute dataset paths
data_config = {
    "path": str(YOLO_DIR),
    "train": "images/train",
    "val": "images/val",
    "names": {
        0: "defect"
    }
}

DATA_YAML = BASE_DIR / "yolo_runtime.yaml"

with open(DATA_YAML, "w") as file:
    yaml.safe_dump(data_config, file, sort_keys=False)

print("Dataset configuration created:", DATA_YAML)

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Train YOLO
results = model.train(
    data=str(DATA_YAML),
    epochs=10,
    imgsz=320,
    batch=4,
    project=str(MODEL_DIR / "yolo_training"),
    name="defect_detector",
    exist_ok=True
)

print("\nYOLO training completed!")

# Load best trained model
best_model_path = (
    MODEL_DIR
    / "yolo_training"
    / "defect_detector"
    / "weights"
    / "best.pt"
)

if best_model_path.exists():
    trained_model = YOLO(str(best_model_path))

    # Test on validation images
    predictions = trained_model.predict(
        source=str(VAL_IMAGES),
        save=True,
        conf=0.25
    )

    print("Predictions completed!")
    print("Best model:", best_model_path)
else:
    print("Training finished, but best.pt was not found.")