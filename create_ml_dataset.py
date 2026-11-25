import numpy as np
import os


print("Loading benign features...")

benign = np.load(
    "data/features/enhanced_mnist_benign.npy"
)

print("Benign shape:", benign.shape)


feature_sets = [benign]
labels = [np.zeros(len(benign), dtype=np.int32)]


for lsb in range(1, 6):

    path = f"data/features/enhanced_mnist_attacked_{lsb}lsb.npy"

    print(f"Loading {lsb}-LSB attacked features...")

    attacked = np.load(path)

    print("Attacked shape:", attacked.shape)

    feature_sets.append(attacked)

    labels.append(
        np.ones(len(attacked), dtype=np.int32)
    )


X = np.vstack(feature_sets)
y = np.concatenate(labels)


print()
print("Final feature dataset:", X.shape)
print("Final labels:", y.shape)

print()
print("Benign samples:", np.sum(y == 0))
print("Malicious samples:", np.sum(y == 1))


os.makedirs("data/features/ml_dataset", exist_ok=True)


np.save(
    "data/features/ml_dataset/X_mnist_enhanced.npy",
    X
)

np.save(
    "data/features/ml_dataset/y_mnist_enhanced.npy",
    y
)


print()
print("ML dataset saved successfully.")