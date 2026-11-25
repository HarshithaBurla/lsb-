import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# LOAD RESULTS
# ============================================================

file_path = "data/features/ml_dataset/final_classifier_results.csv"

df = pd.read_csv(file_path)

classifiers = ["Random Forest", "XGBoost", "MLP"]

# ============================================================
# EXTRACT DATA
# ============================================================

baseline_fpr = []
proposed_fpr = []

baseline_fnr = []
proposed_fnr = []

for classifier in classifiers:

    baseline_row = df[
        (df["Feature_Set"] == "Baseline") &
        (df["Classifier"] == classifier)
    ].iloc[0]

    proposed_row = df[
        (df["Feature_Set"] == "Proposed") &
        (df["Classifier"] == classifier)
    ].iloc[0]

    baseline_fpr.append(baseline_row["FPR"] * 100)
    proposed_fpr.append(proposed_row["FPR"] * 100)

    baseline_fnr.append(baseline_row["FNR"] * 100)
    proposed_fnr.append(proposed_row["FNR"] * 100)


# ============================================================
# GRAPH 1 — FPR
# ============================================================

x = range(len(classifiers))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    [i - width / 2 for i in x],
    baseline_fpr,
    width,
    label="Baseline"
)

plt.bar(
    [i + width / 2 for i in x],
    proposed_fpr,
    width,
    label="Proposed"
)

for i, value in enumerate(baseline_fpr):
    plt.text(
        i - width / 2,
        value + 0.5,
        f"{value:.1f}%",
        ha="center"
    )

for i, value in enumerate(proposed_fpr):
    plt.text(
        i + width / 2,
        value + 0.5,
        f"{value:.1f}%",
        ha="center"
    )

plt.xticks(list(x), classifiers)

plt.xlabel("Classifier")
plt.ylabel("False Positive Rate (%)")

plt.title(
    "Baseline vs Proposed Feature Set: False Positive Rate"
)

plt.ylim(0, 25)

plt.legend()

plt.tight_layout()

fpr_file = (
    "data/features/ml_dataset/"
    "fpr_baseline_vs_proposed.png"
)

plt.savefig(
    fpr_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GRAPH 2 — FNR
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    [i - width / 2 for i in x],
    baseline_fnr,
    width,
    label="Baseline"
)

plt.bar(
    [i + width / 2 for i in x],
    proposed_fnr,
    width,
    label="Proposed"
)

for i, value in enumerate(baseline_fnr):
    plt.text(
        i - width / 2,
        value + 0.5,
        f"{value:.1f}%",
        ha="center"
    )

for i, value in enumerate(proposed_fnr):
    plt.text(
        i + width / 2,
        value + 0.5,
        f"{value:.1f}%",
        ha="center"
    )

plt.xticks(list(x), classifiers)

plt.xlabel("Classifier")
plt.ylabel("False Negative Rate (%)")

plt.title(
    "Baseline vs Proposed Feature Set: False Negative Rate"
)

plt.ylim(0, 30)

plt.legend()

plt.tight_layout()

fnr_file = (
    "data/features/ml_dataset/"
    "fnr_baseline_vs_proposed.png"
)

plt.savefig(
    fnr_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("=" * 70)
print("FPR/FNR GRAPHS CREATED")
print("=" * 70)

print(f"FPR graph: {fpr_file}")
print(f"FNR graph: {fnr_file}")