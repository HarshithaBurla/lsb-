import torch
import numpy as np
import time
import pandas as pd
import os

from data_locator import ret_zoo_weights
from enhanced_features import extract_enhanced_features


# ============================================================
# SETTINGS
# ============================================================

ZOO_NAME = "mnist"
OUTPUT_DIR = "data/features/ml_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD RAW MODEL WEIGHTS
# ============================================================

print("=" * 75)
print("ACTUAL ENHANCED FEATURE EXTRACTION TIME")
print("=" * 75)

print("\nLoading MNIST model weights...")

start_load = time.perf_counter()

weights = ret_zoo_weights(ZOO_NAME)

load_time = time.perf_counter() - start_load

print(f"Raw weights shape: {weights.shape}")
print(f"Number of models: {len(weights)}")
print(f"Weights per model: {weights.shape[1]}")
print(f"Loading time: {load_time:.4f} seconds")


# ============================================================
# WARM-UP
# ============================================================

print("\nRunning warm-up...")

_ = extract_enhanced_features(weights[0])

print("Warm-up completed.")


# ============================================================
# ACTUAL FEATURE EXTRACTION
# ============================================================

print("\nExtracting enhanced features...")

start = time.perf_counter()

all_features = []

for i, model_weights in enumerate(weights):

    features = extract_enhanced_features(model_weights)

    all_features.append(features)

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(weights)} models")


enhanced_features = np.array(all_features, dtype=np.float32)

extraction_time = time.perf_counter() - start


# ============================================================
# CALCULATE METRICS
# ============================================================

number_of_models = len(weights)
number_of_features = enhanced_features.shape[1]

time_per_model = extraction_time / number_of_models

models_per_second = number_of_models / extraction_time


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 75)
print("RESULTS")
print("=" * 75)

print(f"\nFeature matrix shape : {enhanced_features.shape}")

print(f"Features per model  : {number_of_features}")

print(f"Total models        : {number_of_models}")

print(
    f"\nTotal extraction time : "
    f"{extraction_time:.4f} seconds"
)

print(
    f"Time per model        : "
    f"{time_per_model * 1000:.4f} ms"
)

print(
    f"Processing speed      : "
    f"{models_per_second:.2f} models/second"
)


# ============================================================
# SAVE FEATURES
# ============================================================

feature_file = (
    "data/features/enhanced_mnist_benchmark.npy"
)

np.save(feature_file, enhanced_features)

print(
    f"\nFeatures saved to: "
    f"{feature_file}"
)


# ============================================================
# SAVE TIMING RESULT
# ============================================================

results = pd.DataFrame([
    {
        "Dataset": "MNIST",
        "Models": number_of_models,
        "Weights_Per_Model": weights.shape[1],
        "Enhanced_Features": number_of_features,
        "Extraction_Time_Seconds": extraction_time,
        "Time_Per_Model_ms": time_per_model * 1000,
        "Models_Per_Second": models_per_second
    }
])

csv_file = (
    f"{OUTPUT_DIR}/enhanced_feature_extraction_time.csv"
)

results.to_csv(csv_file, index=False)

print(
    f"Timing results saved to: "
    f"{csv_file}"
)

print("\nDONE")