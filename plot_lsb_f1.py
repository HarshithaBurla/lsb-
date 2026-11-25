import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD RESULTS
# ============================================================

file_path = "data/features/ml_dataset/lsb_wise_results.csv"

df = pd.read_csv(file_path)


# ============================================================
# CREATE ONE GRAPH PER CLASSIFIER
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
        data["Baseline_F1"],
        marker="o",
        label="Baseline"
    )

    plt.plot(
        data["LSB"],
        data["Proposed_F1"],
        marker="s",
        label="Proposed"
    )

    plt.xlabel("LSB Attack Level")
    plt.ylabel("F1-Score")

    plt.title(
        f"{classifier}: LSB Attack Level vs F1-Score"
    )

    plt.xticks(range(1, 24))

    plt.ylim(0, 1.05)

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    # Safe filename
    filename = classifier.lower().replace(" ", "_")

    output_file = (
        f"data/features/ml_dataset/"
        f"lsb_f1_{filename}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Saved: {output_file}"
    )


print()
print("=" * 60)
print("F1 GRAPHS CREATED SUCCESSFULLY")
print("=" * 60)