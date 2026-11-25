
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Sample training text
text = (
    "machine learning is interesting "
    "artificial intelligence learns patterns "
    "machine learning builds intelligent systems "
)

# Character vocabulary
chars = sorted(set(text))

char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

# Encode text
encoded = np.array([char_to_id[c] for c in text])

# Create sequences
seq_length = 20

X = []
y = []

for i in range(len(encoded) - seq_length):
    X.append(encoded[i:i + seq_length])
    y.append(encoded[i + seq_length])

X = np.array(X)
y = np.array(y)

# Build RNN
model = models.Sequential([
    layers.Embedding(len(chars), 32),
    layers.SimpleRNN(64),
    layers.Dense(len(chars), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(X, y, epochs=30, batch_size=16)

# Predict next character
sample = X[:1]

prediction = model.predict(sample)
next_char_id = prediction.argmax()

print("Input Sequence:", "".join(id_to_char[i] for i in sample[0]))
print("Predicted Next Character:", id_to_char[next_char_id])
