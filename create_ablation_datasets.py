import numpy as np
import os

# ============================================================
# SETTINGS
# ============================================================

N_BENIGN = 100
N_ATTACK = 20
LSB_LEVELS = [1, 2, 3, 4, 5]

GRAD_FILE = (
    "data/features/grads/"
    "zoo_mnist_malwarename_malware_12584bytes_fill_True_msb_True.npz"
)

LOSS_FILE = (
    "data/features/losses/"
    "zoo_mnist_malwarename_malware_12584bytes_fill_True_msb_True.npz"
)

OUTPUT_DIR = "data/features/ml_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD BASELINE FEATURES
# ============================================================

print("=" * 70)
print("CREATING FEATURE ABLATION DATASETS")
print("=" * 70)

print("\nLoading gradient features...")
grads = np.load(GRAD_FILE)

print("Loading reconstruction losses...")
loss_data = np.load(LOSS_FILE)

# ------------------------------------------------------------
# Load enhanced feature datasets
# ------------------------------------------------------------

print("Loading enhanced feature datasets...")

benign_enhanced = np.load(
    "data/features/enhanced_mnist_benign.npy"
)

attacked_enhanced = {}

for lsb in LSB_LEVELS:
    path = (
        f"data/features/"
        f"enhanced_mnist_attacked_{lsb}lsb.npy"
    )

    attacked_enhanced[lsb] = np.load(path)


# ============================================================
# CREATE DATA
# ============================================================

baseline_list = []
weight_stats_list = []
lsb_features_list = []
proposed_list = []
labels = []

# ------------------------------------------------------------
# BENIGN MODELS
# ------------------------------------------------------------

print("\nProcessing benign models...")

for i in range(N_BENIGN):

    # Gradient features
    gradient = grads["0"][i]

    # Reconstruction loss
    loss = np.array([
        loss_data["losses"][0][i]
    ], dtype=np.float32)

    # Enhanced features
    enhanced = benign_enhanced[i]

    # First 7 = weight statistics
    weight_stats = enhanced[:7]

    # Remaining 15 = LSB features
    lsb_features = enhanced[7:]

    # Baseline = gradient + loss
    baseline = np.concatenate([
        gradient,
        loss
    ])

    # Baseline + weight statistics
    weight_stats_features = np.concatenate([
        baseline,
        weight_stats
    ])

    # Baseline + LSB features
    lsb_features_features = np.concatenate([
        baseline,
        lsb_features
    ])

    # Full proposed features
    proposed = np.concatenate([
        baseline,
        enhanced
    ])

    baseline_list.append(baseline)
    weight_stats_list.append(weight_stats_features)
    lsb_features_list.append(lsb_features_features)
    proposed_list.append(proposed)

    labels.append(0)


# ============================================================
# ATTACKED MODELS
# ============================================================

print("Processing attacked models...")

for lsb in LSB_LEVELS:

    print(f"  Processing {lsb} LSB attack...")

    for i in range(N_ATTACK):

        # Gradient features
        gradient = grads[str(lsb)][i]

        # Reconstruction loss
        # losses row:
        # 0 = benign
        # 1 = 23 LSB
        # ...
        # 23 = 1 LSB
        loss_index = 24 - lsb

        loss = np.array([
            loss_data["losses"][loss_index][i]
        ], dtype=np.float32)

        # Enhanced features
        enhanced = attacked_enhanced[lsb][i]

        # First 7 = weight statistics
        weight_stats = enhanced[:7]

        # Last 15 = LSB features
        lsb_features = enhanced[7:]

        # Baseline
        baseline = np.concatenate([
            gradient,
            loss
        ])

        # Baseline + weight statistics
        weight_stats_features = np.concatenate([
            baseline,
            weight_stats
        ])

        # Baseline + LSB features
        lsb_features_features = np.concatenate([
            baseline,
            lsb_features
        ])

        # Full proposed
        proposed = np.concatenate([
            baseline,
            enhanced
        ])

        baseline_list.append(baseline)
        weight_stats_list.append(weight_stats_features)
        lsb_features_list.append(lsb_features_features)
        proposed_list.append(proposed)

        labels.append(1)


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

X_baseline = np.array(
    baseline_list,
    dtype=np.float32
)

X_weight_stats = np.array(
    weight_stats_list,
    dtype=np.float32
)

X_lsb_features = np.array(
    lsb_features_list,
    dtype=np.float32
)

X_proposed = np.array(
    proposed_list,
    dtype=np.float32
)

y = np.array(
    labels,
    dtype=np.int32
)


# ============================================================
# SHUFFLE ALL DATA IDENTICALLY
# ============================================================

rng = np.random.default_rng(42)

indices = rng.permutation(len(y))

X_baseline = X_baseline[indices]
X_weight_stats = X_weight_stats[indices]
X_lsb_features = X_lsb_features[indices]
X_proposed = X_proposed[indices]
y = y[indices]


# ============================================================
# SAVE
# ============================================================

np.save(
    f"{OUTPUT_DIR}/X_mnist_ablation_baseline.npy",
    X_baseline
)

np.save(
    f"{OUTPUT_DIR}/X_mnist_ablation_weight_stats.npy",
    X_weight_stats
)

np.save(
    f"{OUTPUT_DIR}/X_mnist_ablation_lsb_features.npy",
    X_lsb_features
)

np.save(
    f"{OUTPUT_DIR}/X_mnist_ablation_proposed.npy",
    X_proposed
)

np.save(
    f"{OUTPUT_DIR}/y_mnist_ablation.npy",
    y
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ABLATION DATASETS CREATED")
print("=" * 70)

print(f"Baseline:              {X_baseline.shape}")
print(f"+ Weight Statistics:   {X_weight_stats.shape}")
print(f"+ LSB Features:        {X_lsb_features.shape}")
print(f"Proposed:              {X_proposed.shape}")
print(f"Labels:                {y.shape}")

print("\nClass distribution:")
print(f"Benign:    {np.sum(y == 0)}")
print(f"Malicious: {np.sum(y == 1)}")

print("\nFeature counts:")
print(f"Baseline:            {X_baseline.shape[1]}")
print(f"Weight Statistics:   {X_weight_stats.shape[1]}")
print(f"LSB Features:        {X_lsb_features.shape[1]}")
print(f"Proposed:            {X_proposed.shape[1]}")

print("\nFiles saved successfully.")
print("=" * 70)