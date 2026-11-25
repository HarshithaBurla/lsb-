import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import xgboost as xgb


# ============================================================
# SETTINGS
# ============================================================

BASELINE_FILE = "data/features/ml_dataset/X_mnist_baseline_balanced.npy"
PROPOSED_FILE = "data/features/ml_dataset/X_mnist_proposed_balanced.npy"
LABEL_FILE = "data/features/ml_dataset/y_mnist_comparison.npy"

SEEDS = [42, 7, 21, 100, 123]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

X_baseline = np.load(BASELINE_FILE)
X_proposed = np.load(PROPOSED_FILE)
y = np.load(LABEL_FILE)

print("Baseline shape :", X_baseline.shape)
print("Proposed shape :", X_proposed.shape)
print("Labels shape   :", y.shape)

print()
print("Number of benign samples   :", np.sum(y == 0))
print("Number of malicious samples:", np.sum(y == 1))


# ============================================================
# CLASSIFIER CREATION
# ============================================================

def create_classifiers(seed):

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=-1
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
        eval_metric="logloss",
        n_jobs=-1
    )

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            random_state=seed
        ))
    ])

    return {
        "Random Forest": rf,
        "XGBoost": xgb_model,
        "MLP": mlp
    }


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):

    start_train = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start_train

    start_detect = time.time()

    predictions = model.predict(X_test)

    detect_time = time.time() - start_detect

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = predictions

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "train_time": train_time,
        "detect_time": detect_time
    }


# ============================================================
# STORAGE
# ============================================================

results = {
    "Baseline": {
        "Random Forest": [],
        "XGBoost": [],
        "MLP": []
    },
    "Proposed": {
        "Random Forest": [],
        "XGBoost": [],
        "MLP": []
    }
}


# ============================================================
# RUN 5 EXPERIMENTS
# ============================================================

for seed in SEEDS:

    print()
    print("=" * 70)
    print("SEED:", seed)
    print("=" * 70)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.20,
        stratify=y,
        random_state=seed
    )

    Xb_train = X_baseline[train_idx]
    Xb_test = X_baseline[test_idx]

    Xp_train = X_proposed[train_idx]
    Xp_test = X_proposed[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Training samples:", len(train_idx))
    print("Testing samples :", len(test_idx))

    classifiers = create_classifiers(seed)

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print()
    print("BASELINE")

    for name, model in classifiers.items():

        result = evaluate_model(
            model,
            Xb_train,
            Xb_test,
            y_train,
            y_test
        )

        results["Baseline"][name].append(result)

        print(
            f"{name:16s} "
            f"Accuracy={result['accuracy']:.4f} "
            f"F1={result['f1']:.4f} "
            f"AUC={result['auc']:.4f}"
        )

    # --------------------------------------------------------
    # PROPOSED
    # --------------------------------------------------------

    print()
    print("PROPOSED")

    for name, model in classifiers.items():

        result = evaluate_model(
            model,
            Xp_train,
            Xp_test,
            y_train,
            y_test
        )

        results["Proposed"][name].append(result)

        print(
            f"{name:16s} "
            f"Accuracy={result['accuracy']:.4f} "
            f"F1={result['f1']:.4f} "
            f"AUC={result['auc']:.4f}"
        )


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def get_mean_std(result_list, key):

    values = [
        result[key]
        for result in result_list
    ]

    return np.mean(values), np.std(values)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL 5-SEED SUMMARY")
print("=" * 70)

for method in ["Baseline", "Proposed"]:

    print()
    print("############", method, "############")

    for classifier in [
        "Random Forest",
        "XGBoost",
        "MLP"
    ]:

        data = results[method][classifier]

        acc_mean, acc_std = get_mean_std(data, "accuracy")
        prec_mean, prec_std = get_mean_std(data, "precision")
        rec_mean, rec_std = get_mean_std(data, "recall")
        f1_mean, f1_std = get_mean_std(data, "f1")
        auc_mean, auc_std = get_mean_std(data, "auc")
        fpr_mean, fpr_std = get_mean_std(data, "fpr")
        fnr_mean, fnr_std = get_mean_std(data, "fnr")
        train_mean, train_std = get_mean_std(data, "train_time")
        detect_mean, detect_std = get_mean_std(data, "detect_time")

        print()
        print(classifier)
        print("-" * 50)

        print(
            f"Accuracy : {acc_mean:.4f} ± {acc_std:.4f}"
        )

        print(
            f"Precision: {prec_mean:.4f} ± {prec_std:.4f}"
        )

        print(
            f"Recall   : {rec_mean:.4f} ± {rec_std:.4f}"
        )

        print(
            f"F1-score : {f1_mean:.4f} ± {f1_std:.4f}"
        )

        print(
            f"ROC-AUC  : {auc_mean:.4f} ± {auc_std:.4f}"
        )

        print(
            f"FPR      : {fpr_mean:.4f} ± {fpr_std:.4f}"
        )

        print(
            f"FNR      : {fnr_mean:.4f} ± {fnr_std:.4f}"
        )

        print(
            f"Train time: {train_mean:.4f} ± {train_std:.4f} sec"
        )

        print(
            f"Detect time: {detect_mean:.4f} ± {detect_std:.4f} sec"
        )


# ============================================================
# BASELINE VS PROPOSED
# ============================================================

print()
print("=" * 70)
print("BASELINE vs PROPOSED — MEAN RESULTS")
print("=" * 70)

print()
print(
    f"{'Classifier':16s}"
    f"{'Base Acc':>12s}"
    f"{'Prop Acc':>12s}"
    f"{'Base F1':>12s}"
    f"{'Prop F1':>12s}"
    f"{'Base AUC':>12s}"
    f"{'Prop AUC':>12s}"
)

print("-" * 82)

for classifier in [
    "Random Forest",
    "XGBoost",
    "MLP"
]:

    base_acc, _ = get_mean_std(
        results["Baseline"][classifier],
        "accuracy"
    )

    prop_acc, _ = get_mean_std(
        results["Proposed"][classifier],
        "accuracy"
    )

    base_f1, _ = get_mean_std(
        results["Baseline"][classifier],
        "f1"
    )

    prop_f1, _ = get_mean_std(
        results["Proposed"][classifier],
        "f1"
    )

    base_auc, _ = get_mean_std(
        results["Baseline"][classifier],
        "auc"
    )

    prop_auc, _ = get_mean_std(
        results["Proposed"][classifier],
        "auc"
    )

    print(
        f"{classifier:16s}"
        f"{base_acc:12.4f}"
        f"{prop_acc:12.4f}"
        f"{base_f1:12.4f}"
        f"{prop_f1:12.4f}"
        f"{base_auc:12.4f}"
        f"{prop_auc:12.4f}"
    )


# ============================================================
# ERROR RATE COMPARISON
# ============================================================

print()
print("=" * 70)
print("ERROR RATE COMPARISON — MEAN RESULTS")
print("=" * 70)

print()
print(
    f"{'Classifier':16s}"
    f"{'Base FPR':>12s}"
    f"{'Prop FPR':>12s}"
    f"{'Base FNR':>12s}"
    f"{'Prop FNR':>12s}"
)

print("-" * 64)

for classifier in [
    "Random Forest",
    "XGBoost",
    "MLP"
]:

    base_fpr, _ = get_mean_std(
        results["Baseline"][classifier],
        "fpr"
    )

    prop_fpr, _ = get_mean_std(
        results["Proposed"][classifier],
        "fpr"
    )

    base_fnr, _ = get_mean_std(
        results["Baseline"][classifier],
        "fnr"
    )

    prop_fnr, _ = get_mean_std(
        results["Proposed"][classifier],
        "fnr"
    )

    print(
        f"{classifier:16s}"
        f"{base_fpr:12.4f}"
        f"{prop_fpr:12.4f}"
        f"{base_fnr:12.4f}"
        f"{prop_fnr:12.4f}"
    )


# ============================================================
# FEATURE COUNT
# ============================================================

print()
print("=" * 70)
print("FEATURE COUNT")
print("=" * 70)

print()
print("Baseline features :", X_baseline.shape[1])
print("Proposed features :", X_proposed.shape[1])
print(
    "Additional features in proposed method :",
    X_proposed.shape[1] - X_baseline.shape[1]
)

print()
print("=" * 70)
print("5-SEED EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)