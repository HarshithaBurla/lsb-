import numpy as np
import os


print("Loading benign dataset...")

benign = np.load(
    "data/features/enhanced_mnist_benign.npy"
)

# Use all 5000 benign models
benign = benign[:5000]

X_parts = [benign]
y_parts = [np.zeros(len(benign), dtype=np.int32)]


# Take 1000 samples from each attack level
for lsb in range(1, 6):

    path = f"data/features/enhanced_mnist_attacked_{lsb}lsb.npy"

    print(f"Loading {lsb}-LSB dataset...")

    attacked = np.load(path)

    attacked = attacked[:1000]

    X_parts.append(attacked)
    y_parts.append(
        np.ones(len(attacked), dtype=np.int32)
    )


X = np.vstack(X_parts)
y = np.concatenate(y_parts)


# Shuffle the complete dataset
rng = np.random.default_rng(42)

indices = rng.permutation(len(X))

X = X[indices]
y = y[indices]


print()
print("Final balanced dataset:", X.shape)
print("Labels:", y.shape)
print("Benign:", np.sum(y == 0))
print("Malicious:", np.sum(y == 1))


os.makedirs("data/features/ml_dataset", exist_ok=True)


np.save(
    "data/features/ml_dataset/X_mnist_balanced.npy",
    X
)

np.save(
    "data/features/ml_dataset/y_mnist_balanced.npy",
    y
)


print()
print("Balanced dataset saved successfully.")