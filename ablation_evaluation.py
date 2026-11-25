import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# LOAD DATASETS
# ============================================================

DATA_DIR = "data/features/ml_dataset"

datasets = {
    "Baseline": np.load(
        f"{DATA_DIR}/X_mnist_ablation_baseline.npy"
    ),

    "Weight Statistics": np.load(
        f"{DATA_DIR}/X_mnist_ablation_weight_stats.npy"
    ),

    "LSB Features": np.load(
        f"{DATA_DIR}/X_mnist_ablation_lsb_features.npy"
    ),

    "Proposed": np.load(
        f"{DATA_DIR}/X_mnist_ablation_proposed.npy"
    )
}

y = np.load(
    f"{DATA_DIR}/y_mnist_ablation.npy"
)


# ============================================================
# CLASSIFIERS
# ============================================================

def create_classifier(name, seed):

    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1
        )

    elif name == "XGBoost":
        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
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
                    random_state=seed
                )
            )
        ])


# ============================================================
# EVALUATION
# ============================================================

SEEDS = [42, 7, 21, 100, 123]

results = []


for dataset_name, X in datasets.items():

    print("\n" + "=" * 75)
    print(f"FEATURE SET: {dataset_name}")
    print(f"Shape: {X.shape}")
    print("=" * 75)

    for classifier_name in [
        "Random Forest",
        "XGBoost",
        "MLP"
    ]:

        print(f"\n{classifier_name}")

        accuracy_scores = []
        f1_scores = []
        auc_scores = []

        for seed in SEEDS:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                stratify=y,
                random_state=seed
            )

            model = create_classifier(
                classifier_name,
                seed
            )

            start = time.perf_counter()

            model.fit(
                X_train,
                y_train
            )

            train_time = time.perf_counter() - start

            start = time.perf_counter()

            y_pred = model.predict(X_test)

            detect_time = time.perf_counter() - start

            # Probability scores for ROC-AUC
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )

            auc = roc_auc_score(
                y_test,
                y_prob
            )

            accuracy_scores.append(accuracy)
            f1_scores.append(f1)
            auc_scores.append(auc)

        # ----------------------------------------------------
        # Mean and standard deviation
        # ----------------------------------------------------

        acc_mean = np.mean(accuracy_scores)
        acc_std = np.std(accuracy_scores)

        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores)

        auc_mean = np.mean(auc_scores)
        auc_std = np.std(auc_scores)

        print(
            f"Accuracy : "
            f"{acc_mean:.4f} ± {acc_std:.4f}"
        )

        print(
            f"F1       : "
            f"{f1_mean:.4f} ± {f1_std:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{auc_mean:.4f} ± {auc_std:.4f}"
        )

        results.append([
            dataset_name,
            classifier_name,
            acc_mean,
            acc_std,
            f1_mean,
            f1_std,
            auc_mean,
            auc_std
        ])


# ============================================================
# SAVE RESULTS
# ============================================================

import csv

output_file = (
    f"{DATA_DIR}/ablation_results.csv"
)

with open(
    output_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Feature_Set",
        "Classifier",
        "Accuracy_Mean",
        "Accuracy_STD",
        "F1_Mean",
        "F1_STD",
        "AUC_Mean",
        "AUC_STD"
    ])

    writer.writerows(results)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("ABLATION EXPERIMENT COMPLETED")
print("=" * 75)

print(
    f"Results saved to:\n"
    f"{output_file}"
)

print("=" * 75)