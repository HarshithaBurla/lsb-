import pandas as pd
import matplotlib.pyplot as plt

# Load results
file = "data/features/ml_dataset/ablation_results.csv"

df = pd.read_csv(file)

# Keep MLP only
mlp = df[df["Classifier"] == "MLP"].copy()

# Preserve desired order
order = [
    "Baseline",
    "Weight Statistics",
    "LSB Features",
    "Proposed"
]

mlp["Feature_Set"] = pd.Categorical(
    mlp["Feature_Set"],
    categories=order,
    ordered=True
)

mlp = mlp.sort_values("Feature_Set")

# Plot
plt.figure(figsize=(9, 5))

plt.bar(
    mlp["Feature_Set"],
    mlp["Accuracy_Mean"]
)

plt.ylabel("Accuracy")
plt.xlabel("Feature Configuration")
plt.title("MLP Accuracy for Different Feature Configurations")

plt.ylim(0, 1.0)

plt.xticks(rotation=15)

# Add values above bars
for i, value in enumerate(mlp["Accuracy_Mean"]):

    plt.text(
        i,
        value + 0.02,
        f"{value * 100:.1f}%",
        ha="center"
    )

plt.tight_layout()

output = (
    "data/features/ml_dataset/"
    "ablation_mlp_accuracy.png"
)

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

print("GRAPH SAVED")
print(f"File: {output}")