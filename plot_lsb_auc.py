import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD RESULTS
# ============================================================

file_path = "data/features/ml_dataset/lsb_wise_results.csv"

df = pd.read_csv(file_path)


# ============================================================
# CREATE AUC GRAPHS
# ============================================================

classifiers = [
    "Random Forest",
    "XGBoost",
    "MLP"
]


for classifier in classifiers:

    data = df[df["Classifier"] == classifier]

    plt.figure(figsize=(10, 6))

    plt.plot(
        data["LSB"],
        data["Baseline_AUC"],
        marker="o",
        label="Baseline"
    )

    plt.plot(
        data["LSB"],
        data["Proposed_AUC"],
        marker="s",
        label="Proposed"
    )

    plt.xlabel("LSB Attack Level")
    plt.ylabel("ROC-AUC")

    plt.title(
        f"{classifier}: LSB Attack Level vs ROC-AUC"
    )

    plt.xticks(range(1, 24))

    plt.ylim(0, 1.05)

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    filename = classifier.lower().replace(" ", "_")

    output_file = (
        f"data/features/ml_dataset/"
        f"lsb_auc_{filename}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_file}")


print()
print("=" * 60)
print("ROC-AUC GRAPHS CREATED SUCCESSFULLY")
print("=" * 60)