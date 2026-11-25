import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data/features/ml_dataset"

LSB_LEVELS = range(1, 24)

SEED = 42

N_BENIGN = 100
N_ATTACK = 100


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("FPR / FNR ANALYSIS ACROSS 23 LSB ATTACK LEVELS")
print("=" * 75)

gradient_file = (
    "data/features/grads/"
    "zoo_mnist_malwarename_malware_12584bytes_fill_True_msb_True.npz"
)

loss_file = (
    "data/features/losses/"
    "zoo_mnist_malwarename_malware_12584bytes_fill_True_msb_True.npz"
)

grads = np.load(gradient_file)
losses = np.load(loss_file)

benign_enhanced = np.load(
    "data/features/enhanced_mnist_benign.npy"
)

attacked_enhanced = {}

for lsb in LSB_LEVELS:

    attacked_enhanced[lsb] = np.load(
        f"data/features/"
        f"enhanced_mnist_attacked_{lsb}lsb.npy"
    )


# ============================================================
# CLASSIFIERS
# ============================================================

def create_classifier(name):

    if name == "Random Forest":

        return RandomForestClassifier(
            n_estimators=100,
            random_state=SEED,
            n_jobs=-1
        )

    elif name == "XGBoost":

        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=-1
        )

    elif name == "MLP":

        return Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    max_iter=300,
                    early_stopping=True,
                    random_state=SEED
                )
            )
        ])


# ============================================================
# CREATE RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS EACH LSB LEVEL
# ============================================================

for lsb in LSB_LEVELS:

    print(f"\nProcessing {lsb} LSB...")

    # --------------------------------------------------------
    # BENIGN
    # --------------------------------------------------------

    benign_gradient = grads["0"][:N_BENIGN]

    benign_loss = losses["losses"][0][:N_BENIGN].reshape(-1, 1)

    benign_enh = benign_enhanced[:N_BENIGN]

    benign_baseline = np.concatenate(
        [
            benign_gradient,
            benign_loss
        ],
        axis=1
    )

    benign_proposed = np.concatenate(
        [
            benign_baseline,
            benign_enh
        ],
        axis=1
    )


    # --------------------------------------------------------
    # ATTACKED
    # --------------------------------------------------------

    attacked_gradient = grads[str(lsb)][:N_ATTACK]

    # Loss file ordering:
    # index 0  = benign
    # index 1  = 23 LSB
    # ...
    # index 23 = 1 LSB

    loss_index = 24 - lsb

    attacked_loss = (
        losses["losses"][loss_index][:N_ATTACK]
        .reshape(-1, 1)
    )

    attacked_enh = attacked_enhanced[lsb][:N_ATTACK]

    attacked_baseline = np.concatenate(
        [
            attacked_gradient,
            attacked_loss
        ],
        axis=1
    )

    attacked_proposed = np.concatenate(
        [
            attacked_baseline,
            attacked_enh
        ],
        axis=1
    )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    X_baseline = np.concatenate(
        [
            benign_baseline,
            attacked_baseline
        ],
        axis=0
    )

    X_proposed = np.concatenate(
        [
            benign_proposed,
            attacked_proposed
        ],
        axis=0
    )

    y = np.concatenate(
        [
            np.zeros(N_BENIGN),
            np.ones(N_ATTACK)
        ]
    )


    # --------------------------------------------------------
    # SAME TRAIN/TEST SPLIT
    # --------------------------------------------------------

    indices = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=SEED
    )


    # ========================================================
    # CLASSIFIERS
    # ========================================================

    for classifier_name in [
        "Random Forest",
        "XGBoost",
        "MLP"
    ]:

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        model = create_classifier(classifier_name)

        model.fit(
            X_baseline[train_idx],
            y[train_idx]
        )

        pred = model.predict(
            X_baseline[test_idx]
        )

        tn, fp, fn, tp = confusion_matrix(
            y[test_idx],
            pred
        ).ravel()

        baseline_fpr = fp / (fp + tn)

        baseline_fnr = fn / (fn + tp)


        # ----------------------------------------------------
        # PROPOSED
        # ----------------------------------------------------

        model = create_classifier(classifier_name)

        model.fit(
            X_proposed[train_idx],
            y[train_idx]
        )

        pred = model.predict(
            X_proposed[test_idx]
        )

        tn, fp, fn, tp = confusion_matrix(
            y[test_idx],
            pred
        ).ravel()

        proposed_fpr = fp / (fp + tn)

        proposed_fnr = fn / (fn + tp)


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        results.append([
            classifier_name,
            lsb,
            baseline_fpr,
            proposed_fpr,
            baseline_fnr,
            proposed_fnr
        ])


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(
    results,
    columns=[
        "Classifier",
        "LSB",
        "Baseline_FPR",
        "Proposed_FPR",
        "Baseline_FNR",
        "Proposed_FNR"
    ]
)

output_file = (
    f"{DATA_DIR}/lsb_fpr_fnr_results.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FPR / FNR ANALYSIS COMPLETED")
print("=" * 75)

print("\nAverage across 23 LSB levels:")

for classifier in [
    "Random Forest",
    "XGBoost",
    "MLP"
]:

    data = df[
        df["Classifier"] == classifier
    ]

    print(f"\n{classifier}")

    print(
        f"Baseline FPR : "
        f"{data['Baseline_FPR'].mean():.4f}"
    )

    print(
        f"Proposed FPR : "
        f"{data['Proposed_FPR'].mean():.4f}"
    )

    print(
        f"Baseline FNR : "
        f"{data['Baseline_FNR'].mean():.4f}"
    )

    print(
        f"Proposed FNR : "
        f"{data['Proposed_FNR'].mean():.4f}"
    )

print("\n" + "-" * 75)
print("RESULTS SAVED")
print("-" * 75)
print(f"File: {output_file}")
print(f"Rows: {len(df)}")