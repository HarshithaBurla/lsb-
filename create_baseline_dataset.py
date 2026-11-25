import numpy as np
import os

print("Loading gradient features...")

grad_path = (
    "data/features/grads/"
    "zoo_mnist_malwarename_malware_12584bytes_fill_True_msb_True.npz"
)

grad_data = np.load(grad_path)

print("Gradient arrays:", grad_data.files)

# Use only 100 models for this baseline experiment
N = 100

feature_parts = []
labels = []

# Benign
benign_grad = grad_data["0"][:N]

print("Benign gradient shape:", benign_grad.shape)

# Reconstruction losses
loss_path = (
    "data/features/losses/"
    "zoo_mnist_malwarename_malware_12584bytes_fill_True_msb_True.npz"
)

loss_data = np.load(loss_path)

losses = loss_data["losses"]

print("Loss shape:", losses.shape)

# -------------------------------------------------
# Benign models
# -------------------------------------------------

benign_loss = losses[0][:N].reshape(-1, 1)

benign_features = np.hstack([
    benign_grad,
    benign_loss
])

feature_parts.append(benign_features)
labels.append(np.zeros(N, dtype=np.int32))

# -------------------------------------------------
# Attacked models: 1 to 5 LSB
# -------------------------------------------------

for lsb in range(1, 6):

    print(f"Processing {lsb}-LSB...")

    attacked_grad = grad_data[str(lsb)][:N]

    attacked_loss = losses[lsb][:N].reshape(-1, 1)

    attacked_features = np.hstack([
        attacked_grad,
        attacked_loss
    ])

    feature_parts.append(attacked_features)

    labels.append(
        np.ones(N, dtype=np.int32)
    )

# -------------------------------------------------
# Combine
# -------------------------------------------------

X = np.vstack(feature_parts)
y = np.concatenate(labels)

# Shuffle
rng = np.random.default_rng(42)

indices = rng.permutation(len(X))

X = X[indices]
y = y[indices]

print()
print("==============================")
print("BASELINE DATASET")
print("==============================")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Benign:", np.sum(y == 0))
print("Malicious:", np.sum(y == 1))

# Save
os.makedirs(
    "data/features/ml_dataset",
    exist_ok=True
)

np.save(
    "data/features/ml_dataset/X_mnist_baseline.npy",
    X
)

np.save(
    "data/features/ml_dataset/y_mnist_baseline.npy",
    y
)

print()
print("Baseline dataset saved successfully.")