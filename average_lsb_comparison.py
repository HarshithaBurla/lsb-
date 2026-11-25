import pandas as pd


# ============================================================
# LOAD LSB-WISE RESULTS
# ============================================================

file_path = "data/features/ml_dataset/lsb_wise_results.csv"

df = pd.read_csv(file_path)


# ============================================================
# CALCULATE AVERAGES
# ============================================================

summary = (
    df.groupby("Classifier")
    .agg(
        Baseline_Accuracy=("Baseline_Accuracy", "mean"),
        Proposed_Accuracy=("Proposed_Accuracy", "mean"),

        Baseline_F1=("Baseline_F1", "mean"),
        Proposed_F1=("Proposed_F1", "mean"),

        Baseline_AUC=("Baseline_AUC", "mean"),
        Proposed_AUC=("Proposed_AUC", "mean")
    )
    .reset_index()
)


# ============================================================
# CALCULATE IMPROVEMENT
# ============================================================

summary["Accuracy_Change"] = (
    summary["Proposed_Accuracy"]
    - summary["Baseline_Accuracy"]
)

summary["F1_Change"] = (
    summary["Proposed_F1"]
    - summary["Baseline_F1"]
)

summary["AUC_Change"] = (
    summary["Proposed_AUC"]
    - summary["Baseline_AUC"]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 90)
print("AVERAGE PERFORMANCE ACROSS 23 LSB ATTACK LEVELS")
print("=" * 90)

print()

for _, row in summary.iterrows():

    print(f"Classifier: {row['Classifier']}")

    print(
        f"  Accuracy : "
        f"{row['Baseline_Accuracy']:.4f} -> "
        f"{row['Proposed_Accuracy']:.4f} "
        f"(change {row['Accuracy_Change']:+.4f})"
    )

    print(
        f"  F1       : "
        f"{row['Baseline_F1']:.4f} -> "
        f"{row['Proposed_F1']:.4f} "
        f"(change {row['F1_Change']:+.4f})"
    )

    print(
        f"  ROC-AUC  : "
        f"{row['Baseline_AUC']:.4f} -> "
        f"{row['Proposed_AUC']:.4f} "
        f"(change {row['AUC_Change']:+.4f})"
    )

    print("-" * 60)


# ============================================================
# SAVE CSV
# ============================================================

output_file = (
    "data/features/ml_dataset/"
    "lsb_average_comparison.csv"
)

summary.to_csv(
    output_file,
    index=False
)


print()
print("=" * 90)
print("SUMMARY SAVED")
print("=" * 90)

print("File:", output_file)