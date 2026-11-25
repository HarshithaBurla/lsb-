import pandas as pd
import matplotlib.pyplot as plt

# Load results
file_path = "data/features/ml_dataset/final_classifier_results.csv"
df = pd.read_csv(file_path)

classifiers = ["Random Forest", "XGBoost", "MLP"]

baseline = []
proposed = []

for classifier in classifiers:

    b = df[
        (df["Feature_Set"] == "Baseline") &
        (df["Classifier"] == classifier)
    ]["ROC_AUC"].iloc[0]

    p = df[
        (df["Feature_Set"] == "Proposed") &
        (df["Classifier"] == classifier)
    ]["ROC_AUC"].iloc[0]

    baseline.append(b * 100)
    proposed.append(p * 100)


# Create graph
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

# Add values
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

plt.xticks(list(x), classifiers)

plt.xlabel("Classifier")
plt.ylabel("ROC-AUC (%)")

plt.title(
    "Baseline vs Proposed Feature Set: ROC-AUC Comparison"
)

plt.ylim(0, 110)

plt.legend()

plt.tight_layout()

output_file = (
    "data/features/ml_dataset/"
    "roc_auc_baseline_vs_proposed.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("=" * 70)
print("ROC-AUC GRAPH CREATED")
print("=" * 70)
print(f"Saved to: {output_file}")