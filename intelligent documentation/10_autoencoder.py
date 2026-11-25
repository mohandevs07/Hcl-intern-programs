
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Load MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# Encoder
encoder = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(32, activation="relu")
])

# Decoder
decoder = models.Sequential([
    layers.Input(shape=(32,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(784, activation="sigmoid")
])

# Autoencoder
autoencoder = models.Sequential([
    encoder,
    decoder
])

autoencoder.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# Train to reconstruct input
autoencoder.fit(
    x_train,
    x_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, x_test)
)

# Generate reconstructed images
reconstructed = autoencoder.predict(x_test[:10])

# Display original and reconstructed images
plt.figure(figsize=(12, 4))

for i in range(10):

    # Original
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

    # Reconstructed
    plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

plt.suptitle("Original vs Reconstructed Images")
plt.show()

# Demonstrate dimensionality reduction
compressed = encoder.predict(x_test[:10])

print("Original Shape:", x_test[:10].shape)
print("Compressed Shape:", compressed.shape)
