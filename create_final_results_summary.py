import pandas as pd
import os

OUTPUT_DIR = "data/features/ml_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. MAIN CLASSIFIER RESULTS
# ============================================================

main_results = [
    # Baseline
    ["Baseline", "Random Forest", 0.9950, 0.9905, 1.0000, 0.9951,
     1.0000, 0.0100, 0.0000, 0.1261, 0.0157],

    ["Baseline", "XGBoost", 1.0000, 1.0000, 1.0000, 1.0000,
     1.0000, 0.0000, 0.0000, 0.0412, 0.0012],

    ["Baseline", "MLP", 0.8200, 0.8729, 0.7800, 0.8174,
     0.9115, 0.1400, 0.2200, 0.0452, 0.0004],

    # Proposed
    ["Proposed", "Random Forest", 0.9900, 0.9818, 1.0000, 0.9905,
     1.0000, 0.0200, 0.0000, 0.1271, 0.0154],

    ["Proposed", "XGBoost", 0.9900, 0.9818, 1.0000, 0.9905,
     1.0000, 0.0200, 0.0000, 0.0402, 0.0010],

    ["Proposed", "MLP", 0.8550, 0.8813, 0.8200, 0.8494,
     0.9410, 0.1100, 0.1800, 0.0427, 0.0009],
]

columns = [
    "Feature_Set",
    "Classifier",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "FPR",
    "FNR",
    "Training_Time_s",
    "Detection_Time_s"
]

df_main = pd.DataFrame(main_results, columns=columns)


# ============================================================
# 2. FEATURE INFORMATION
# ============================================================

feature_info = pd.DataFrame([
    ["Baseline", 211, 210, 0, 1],
    ["Proposed", 233, 210, 22, 1],
], columns=[
    "Feature_Set",
    "Total_Features",
    "Gradient_Features",
    "Enhanced_Features",
    "Reconstruction_Loss_Features"
])


# ============================================================
# 3. ACTUAL ENHANCED EXTRACTION TIME
# ============================================================

extraction_info = pd.DataFrame([
    [
        "MNIST",
        5000,
        2464,
        22,
        1.0294,
        0.2059,
        4857.31
    ]
], columns=[
    "Dataset",
    "Models",
    "Weights_Per_Model",
    "Enhanced_Features",
    "Extraction_Time_s",
    "Time_Per_Model_ms",
    "Models_Per_Second"
])


# ============================================================
# 4. MERGE MAIN + FEATURE INFORMATION
# ============================================================

final_table = df_main.merge(
    feature_info,
    on="Feature_Set",
    how="left"
)


# ============================================================
# 5. SAVE MAIN MASTER TABLE
# ============================================================

main_file = f"{OUTPUT_DIR}/final_classifier_results.csv"

final_table.to_csv(
    main_file,
    index=False
)


# ============================================================
# 6. SAVE FEATURE INFORMATION
# ============================================================

feature_file = f"{OUTPUT_DIR}/final_feature_information.csv"

feature_info.to_csv(
    feature_file,
    index=False
)


# ============================================================
# 7. SAVE EXTRACTION INFORMATION
# ============================================================

extraction_file = (
    f"{OUTPUT_DIR}/final_extraction_efficiency.csv"
)

extraction_info.to_csv(
    extraction_file,
    index=False
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print("=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)

print("\nMAIN CLASSIFIER RESULTS\n")

print(
    final_table.to_string(index=False)
)

print("\n\nFEATURE INFORMATION\n")

print(
    feature_info.to_string(index=False)
)

print("\n\nEXTRACTION EFFICIENCY\n")

print(
    extraction_info.to_string(index=False)
)


# ============================================================
# 9. PRINT IMPORTANT CHANGES
# ============================================================

baseline_mlp = df_main[
    (df_main["Feature_Set"] == "Baseline") &
    (df_main["Classifier"] == "MLP")
].iloc[0]

proposed_mlp = df_main[
    (df_main["Feature_Set"] == "Proposed") &
    (df_main["Classifier"] == "MLP")
].iloc[0]

print("\n" + "=" * 80)
print("MLP IMPROVEMENT")
print("=" * 80)

print(
    f"Accuracy : "
    f"{baseline_mlp['Accuracy']:.4f} -> "
    f"{proposed_mlp['Accuracy']:.4f}"
)

print(
    f"F1       : "
    f"{baseline_mlp['F1']:.4f} -> "
    f"{proposed_mlp['F1']:.4f}"
)

print(
    f"ROC-AUC  : "
    f"{baseline_mlp['ROC_AUC']:.4f} -> "
    f"{proposed_mlp['ROC_AUC']:.4f}"
)

print(
    f"FPR      : "
    f"{baseline_mlp['FPR']:.4f} -> "
    f"{proposed_mlp['FPR']:.4f}"
)

print(
    f"FNR      : "
    f"{baseline_mlp['FNR']:.4f} -> "
    f"{proposed_mlp['FNR']:.4f}"
)

print("\nFiles saved:")
print(main_file)
print(feature_file)
print(extraction_file)

print("\nDONE")