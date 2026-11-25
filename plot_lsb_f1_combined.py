import pandas as pd
import matplotlib.pyplot as plt

# Load results
file_path = "data/features/ml_dataset/lsb_wise_results.csv"
df = pd.read_csv(file_path)

classifiers = ["Random Forest", "XGBoost", "MLP"]

fig, axes = plt.subplots(3, 1, figsize=(10, 13))

for ax, classifier in zip(axes, classifiers):

    data = df[df["Classifier"] == classifier].sort_values("LSB")

    ax.plot(
        data["LSB"],
        data["Baseline_F1"],
        marker="o",
        label="Baseline"
    )

    ax.plot(
        data["LSB"],
        data["Proposed_F1"],
        marker="s",
        label="Proposed"
    )

    ax.set_title(f"{classifier}: F1-score across LSB attack levels")
    ax.set_xlabel("LSB Attack Level")
    ax.set_ylabel("F1-score")
    ax.set_xticks(range(1, 24))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle(
    "Baseline vs Proposed Feature Set: LSB-wise F1-score",
    fontsize=14
)

plt.tight_layout()

output_file = (
    "data/features/ml_dataset/"
    "lsb_f1_combined.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 70)
print("LSB-WISE F1 GRAPH CREATED")
print("=" * 70)
print(f"Saved to: {output_file}")