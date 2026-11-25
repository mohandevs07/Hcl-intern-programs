import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build MLP model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Convert 28x28 image to 784 features
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 output classes
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")