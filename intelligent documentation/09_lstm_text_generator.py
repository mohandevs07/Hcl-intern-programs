
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Training text
text = (
    "artificial intelligence is changing technology "
    "machine learning helps computers learn patterns "
    "deep learning uses neural networks "
    "artificial intelligence machine learning "
)

# Character vocabulary
chars = sorted(set(text))

char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

# Encode text
encoded = np.array([char_to_id[c] for c in text])

# Create training sequences
seq_length = 30

X = []
y = []

for i in range(len(encoded) - seq_length):
    X.append(encoded[i:i + seq_length])
    y.append(encoded[i + seq_length])

X = np.array(X)
y = np.array(y)

# Build LSTM
model = models.Sequential([
    layers.Embedding(len(chars), 64),
    layers.LSTM(128),
    layers.Dense(len(chars), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

# Train
model.fit(X, y, epochs=30, batch_size=16)

# Generate text
seed = "artificial intelligence"
generated = seed

for _ in range(100):
    sequence = generated[-seq_length:]

    sequence = np.array([
        char_to_id.get(c, 0) for c in sequence
    ])

    sequence = sequence.reshape(1, -1)

    prediction = model.predict(sequence, verbose=0)

    # Random sampling
    probabilities = prediction[0]

    next_id = np.random.choice(
        len(chars),
        p=probabilities / probabilities.sum()
    )

    generated += id_to_char[next_id]

print("\nGenerated Text:")
print(generated)
