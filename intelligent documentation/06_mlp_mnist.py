
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build MLP
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)

# Predict one digit
prediction = model.predict(x_test[:1])
predicted_digit = prediction.argmax()

print("Predicted Digit:", predicted_digit)
print("Actual Digit:", y_test[0])

plt.imshow(x_test[0], cmap="gray")
plt.title(f"Predicted: {predicted_digit}")
plt.axis("off")
plt.show()
