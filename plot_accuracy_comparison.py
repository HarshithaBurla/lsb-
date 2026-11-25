import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# LOAD RESULTS
# ============================================================

file_path = "data/features/ml_dataset/final_classifier_results.csv"

df = pd.read_csv(file_path)

# ============================================================
# PREPARE DATA
# ============================================================

classifiers = ["Random Forest", "XGBoost", "MLP"]

baseline = []
proposed = []

for classifier in classifiers:

    baseline_value = df[
        (df["Feature_Set"] == "Baseline") &
        (df["Classifier"] == classifier)
    ]["Accuracy"].iloc[0]

    proposed_value = df[
        (df["Feature_Set"] == "Proposed") &
        (df["Classifier"] == classifier)
    ]["Accuracy"].iloc[0]

    baseline.append(baseline_value * 100)
    proposed.append(proposed_value * 100)


# ============================================================
# CREATE GRAPH
# ============================================================

x = range(len(classifiers))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    [i - width / 2 for i in x],
    baseline,
    width,
    label="Baseline"
)

plt.bar(
    [i + width / 2 for i in x],
    proposed,
    width,
    label="Proposed"
)

# Add values above bars
for i, value in enumerate(baseline):
    plt.text(
        i - width / 2,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

for i, value in enumerate(proposed):
    plt.text(
        i + width / 2,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

plt.xticks(
    list(x),
    classifiers
)

plt.ylabel("Accuracy (%)")

plt.xlabel("Classifier")

plt.title(
    "Baseline vs Proposed Feature Set: Accuracy Comparison"
)

plt.ylim(0, 110)

plt.legend()

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_file = (
    "data/features/ml_dataset/"
    "accuracy_baseline_vs_proposed.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 70)
print("ACCURACY GRAPH CREATED")
print("=" * 70)
print(f"Saved to: {output_file}")