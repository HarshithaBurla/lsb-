import numpy as np
import time
import pandas as pd
import os

# ============================================================
# SETTINGS
# ============================================================

N_MODELS = 100

GRAD_FILE = (
    "data/features/grads/"
    "zoo_mnist_malwarename_malware_12584bytes_fill_True_msb_True.npz"
)

LOSS_FILE = (
    "data/features/losses/"
    "zoo_mnist_malwarename_malware_12584bytes_fill_True_msb_True.npz"
)

ENHANCED_FILE = (
    "data/features/enhanced_mnist_benign.npy"
)

OUTPUT_DIR = "data/features/ml_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("FEATURE EXTRACTION TIME ANALYSIS")
print("=" * 75)

print("\nLoading existing feature data...")

grads = np.load(GRAD_FILE)
losses = np.load(LOSS_FILE)
enhanced = np.load(ENHANCED_FILE)

# Use the same 5000 MNIST models
grad_features = grads["0"][:N_MODELS]
loss_features = losses["losses"][0][:N_MODELS].reshape(-1, 1)
enhanced_features = enhanced[:N_MODELS]

print(f"Gradient features : {grad_features.shape}")
print(f"Loss features     : {loss_features.shape}")
print(f"Enhanced features : {enhanced_features.shape}")


# ============================================================
# BASELINE FEATURE EXTRACTION
# ============================================================

print("\nMeasuring baseline feature preparation...")

start = time.perf_counter()

baseline_features = np.concatenate(
    [
        grad_features,
        loss_features
    ],
    axis=1
)

baseline_time = time.perf_counter() - start


# ============================================================
# PROPOSED FEATURE EXTRACTION
# ============================================================

print("Measuring proposed feature preparation...")

start = time.perf_counter()

proposed_features = np.concatenate(
    [
        grad_features,
        loss_features,
        enhanced_features
    ],
    axis=1
)

proposed_time = time.perf_counter() - start


# ============================================================
# CALCULATE TIMES
# ============================================================

baseline_per_model = baseline_time / N_MODELS

proposed_per_model = proposed_time / N_MODELS

additional_time = proposed_time - baseline_time

additional_per_model = (
    additional_time / N_MODELS
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("RESULTS")
print("=" * 75)

print(
    f"\nNumber of models: {N_MODELS}"
)

print(
    f"\nBaseline feature count : "
    f"{baseline_features.shape[1]}"
)

print(
    f"Proposed feature count : "
    f"{proposed_features.shape[1]}"
)

print(
    f"\nBaseline preparation time : "
    f"{baseline_time:.6f} seconds"
)

print(
    f"Proposed preparation time : "
    f"{proposed_time:.6f} seconds"
)

print(
    f"\nBaseline time/model : "
    f"{baseline_per_model * 1000:.6f} ms"
)

print(
    f"Proposed time/model : "
    f"{proposed_per_model * 1000:.6f} ms"
)

print(
    f"\nAdditional time for enhanced features : "
    f"{additional_time:.6f} seconds"
)

print(
    f"Additional time/model : "
    f"{additional_per_model * 1000:.6f} ms"
)


# ============================================================
# SAVE CSV
# ============================================================

results = pd.DataFrame([
    {
        "Feature_Set": "Baseline",
        "Number_of_Features": baseline_features.shape[1],
        "Number_of_Models": N_MODELS,
        "Total_Time_Seconds": baseline_time,
        "Time_Per_Model_ms": baseline_per_model * 1000
    },
    {
        "Feature_Set": "Proposed",
        "Number_of_Features": proposed_features.shape[1],
        "Number_of_Models": N_MODELS,
        "Total_Time_Seconds": proposed_time,
        "Time_Per_Model_ms": proposed_per_model * 1000
    }
])

output_file = (
    f"{OUTPUT_DIR}/feature_extraction_time.csv"
)

results.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 75)
print("RESULTS SAVED")
print("=" * 75)

print(f"File: {output_file}")