
import tensorflow as tf
from tensorflow.keras import layers, models

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Add channel dimension and normalize
x_train = x_train[..., None] / 255.0
x_test = x_test[..., None] / 255.0

# Build CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)

print("CNN Test Accuracy:", accuracy)

# Predict
prediction = model.predict(x_test[:1])
print("Predicted Digit:", prediction.argmax())
print("Actual Digit:", y_test[0])
