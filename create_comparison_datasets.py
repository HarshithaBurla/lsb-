import numpy as np
import os

print("Loading datasets...")

# -------------------------------------------------
# Paths
# -------------------------------------------------

grad_path = (
    "data/features/grads/"
    "zoo_mnist_malwarename_malware_12584bytes_fill_True_msb_True.npz"
)

loss_path = (
    "data/features/losses/"
    "zoo_mnist_malwarename_malware_12584bytes_fill_True_msb_True.npz"
)

grad_data = np.load(grad_path)
loss_data = np.load(loss_path)

print("Gradient data loaded.")
print("Loss data loaded.")

# Enhanced feature datasets
enhanced_benign = np.load(
    "data/features/enhanced_mnist_benign.npy"
)

# -------------------------------------------------
# Storage
# -------------------------------------------------

baseline_parts = []
proposed_parts = []

labels = []

# -------------------------------------------------
# 1. BENIGN
# -------------------------------------------------

print()
print("Processing benign models...")

N_BENIGN = 100

benign_grad = grad_data["0"][:N_BENIGN]
benign_loss = loss_data["losses"][0][:N_BENIGN].reshape(-1, 1)
benign_enhanced = enhanced_benign[:N_BENIGN]

# Baseline = Gradient + Loss
benign_baseline = np.hstack([
    benign_grad,
    benign_loss
])

# Proposed = Gradient + Loss + Enhanced
benign_proposed = np.hstack([
    benign_grad,
    benign_loss,
    benign_enhanced
])

baseline_parts.append(benign_baseline)
proposed_parts.append(benign_proposed)

labels.append(
    np.zeros(N_BENIGN, dtype=np.int32)
)

print("Benign baseline:", benign_baseline.shape)
print("Benign proposed:", benign_proposed.shape)

# -------------------------------------------------
# 2. ATTACKED MODELS
# -------------------------------------------------

N_ATTACK = 20

for lsb in range(1, 6):

    print()
    print(f"Processing {lsb}-LSB attacked models...")

    # Gradient features
    attacked_grad = grad_data[str(lsb)][:N_ATTACK]

    # Reconstruction loss
    attacked_loss = (
        loss_data["losses"][lsb][:N_ATTACK]
        .reshape(-1, 1)
    )

    # Enhanced features
    enhanced_path = (
        f"data/features/"
        f"enhanced_mnist_attacked_{lsb}lsb.npy"
    )

    attacked_enhanced = np.load(
        enhanced_path
    )[:N_ATTACK]

    # ---------------------------------------------
    # Baseline
    # ---------------------------------------------

    attacked_baseline = np.hstack([
        attacked_grad,
        attacked_loss
    ])

    # ---------------------------------------------
    # Proposed
    # ---------------------------------------------

    attacked_proposed = np.hstack([
        attacked_grad,
        attacked_loss,
        attacked_enhanced
    ])

    baseline_parts.append(
        attacked_baseline
    )

    proposed_parts.append(
        attacked_proposed
    )

    labels.append(
        np.ones(N_ATTACK, dtype=np.int32)
    )

    print(
        "Baseline shape:",
        attacked_baseline.shape
    )

    print(
        "Proposed shape:",
        attacked_proposed.shape
    )

# -------------------------------------------------
# 3. COMBINE
# -------------------------------------------------

X_baseline = np.vstack(
    baseline_parts
)

X_proposed = np.vstack(
    proposed_parts
)

y = np.concatenate(
    labels
)

# -------------------------------------------------
# 4. SHUFFLE
# -------------------------------------------------

rng = np.random.default_rng(42)

indices = rng.permutation(
    len(y)
)

X_baseline = X_baseline[indices]
X_proposed = X_proposed[indices]
y = y[indices]

# -------------------------------------------------
# 5. DISPLAY
# -------------------------------------------------

print()
print("==============================")
print("COMPARISON DATASETS")
print("==============================")

print()
print("Baseline dataset:")
print("Shape:", X_baseline.shape)

print()
print("Proposed dataset:")
print("Shape:", X_proposed.shape)

print()
print("Labels:")
print("Shape:", y.shape)

print()
print("Benign:", np.sum(y == 0))
print("Malicious:", np.sum(y == 1))

# -------------------------------------------------
# 6. SAVE
# -------------------------------------------------

output_dir = (
    "data/features/ml_dataset"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

np.save(
    f"{output_dir}/X_mnist_baseline_balanced.npy",
    X_baseline
)

np.save(
    f"{output_dir}/X_mnist_proposed_balanced.npy",
    X_proposed
)

np.save(
    f"{output_dir}/y_mnist_comparison.npy",
    y
)

print()
print("Datasets saved successfully.")

print(
    "Baseline:",
    f"{output_dir}/X_mnist_baseline_balanced.npy"
)

print(
    "Proposed:",
    f"{output_dir}/X_mnist_proposed_balanced.npy"
)

print(
    "Labels:",
    f"{output_dir}/y_mnist_comparison.npy"
)