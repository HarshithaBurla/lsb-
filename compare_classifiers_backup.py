import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading datasets...")

X_baseline = np.load(
    "data/features/ml_dataset/X_mnist_baseline_balanced.npy"
)

X_proposed = np.load(
    "data/features/ml_dataset/X_mnist_proposed_balanced.npy"
)

y = np.load(
    "data/features/ml_dataset/y_mnist_comparison.npy"
)

print("Baseline shape :", X_baseline.shape)
print("Proposed shape :", X_proposed.shape)
print("Labels shape   :", y.shape)


# ============================================================
# 2. CHECK DATA
# ============================================================

print()
print("Number of benign samples   :", np.sum(y == 0))
print("Number of malicious samples:", np.sum(y == 1))


# ============================================================
# 3. SAME TRAIN/TEST SPLIT
# ============================================================

indices = np.arange(len(y))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

Xb_train = X_baseline[train_idx]
Xb_test = X_baseline[test_idx]

Xp_train = X_proposed[train_idx]
Xp_test = X_proposed[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

print()
print("Training samples:", len(train_idx))
print("Testing samples :", len(test_idx))

print()
print("Baseline training shape:", Xb_train.shape)
print("Baseline testing shape :", Xb_test.shape)

print()
print("Proposed training shape:", Xp_train.shape)
print("Proposed testing shape :", Xp_test.shape)


# ============================================================
# 4. CLASSIFIERS
# ============================================================

def get_classifiers():

    classifiers = {}

    classifiers["Random Forest"] = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    classifiers["XGBoost"] = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    classifiers["MLP"] = Pipeline([
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
                random_state=42
            )
        )
    ])

    return classifiers


# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    print()
    print("========================================")
    print(name)
    print("========================================")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    start_train = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    end_train = time.perf_counter()

    training_time = end_train - start_train

    # --------------------------------------------------------
    # Detection / Prediction
    # --------------------------------------------------------

    start_detection = time.perf_counter()

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    end_detection = time.perf_counter()

    detection_time = end_detection - start_detection

    # --------------------------------------------------------
    # Classification Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

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

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # FPR and FNR
    # --------------------------------------------------------

    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0.0

    if (fn + tp) > 0:
        fnr = fn / (fn + tp)
    else:
        fnr = 0.0

    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------

    print()
    print("Accuracy       :", f"{accuracy:.4f}")
    print("Precision      :", f"{precision:.4f}")
    print("Recall         :", f"{recall:.4f}")
    print("F1-score       :", f"{f1:.4f}")
    print("ROC-AUC        :", f"{roc_auc:.4f}")

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print("True Negative  :", tn)
    print("False Positive :", fp)
    print("False Negative :", fn)
    print("True Positive  :", tp)

    print()
    print("False Positive Rate:", f"{fpr:.4f}")
    print("False Negative Rate:", f"{fnr:.4f}")

    print()
    print("Training time  :", f"{training_time:.4f}", "seconds")
    print("Detection time :", f"{detection_time:.4f}", "seconds")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "training_time": training_time,
        "detection_time": detection_time
    }


# ============================================================
# 6. RUN BASELINE CLASSIFIERS
# ============================================================

print()
print()
print("############################################################")
print("#                    BASELINE METHOD                       #")
print("############################################################")

baseline_results = {}

baseline_classifiers = get_classifiers()

for name, model in baseline_classifiers.items():

    baseline_results[name] = evaluate_model(
        name,
        model,
        Xb_train,
        Xb_test,
        y_train,
        y_test
    )


# ============================================================
# 7. RUN PROPOSED CLASSIFIERS
# ============================================================

print()
print()
print("############################################################")
print("#                    PROPOSED METHOD                       #")
print("############################################################")

proposed_results = {}

proposed_classifiers = get_classifiers()

for name, model in proposed_classifiers.items():

    proposed_results[name] = evaluate_model(
        name,
        model,
        Xp_train,
        Xp_test,
        y_train,
        y_test
    )


# ============================================================
# 8. FINAL ACCURACY AND F1 COMPARISON
# ============================================================

print()
print()
print("============================================================")
print("                 BASELINE vs PROPOSED")
print("============================================================")

print()

print(
    f"{'Classifier':<18}"
    f"{'Base Acc':<12}"
    f"{'Prop Acc':<12}"
    f"{'Base F1':<12}"
    f"{'Prop F1':<12}"
)

print("-" * 66)

for name in baseline_results:

    base = baseline_results[name]
    prop = proposed_results[name]

    print(
        f"{name:<18}"
        f"{base['accuracy']:<12.4f}"
        f"{prop['accuracy']:<12.4f}"
        f"{base['f1']:<12.4f}"
        f"{prop['f1']:<12.4f}"
    )


# ============================================================
# 9. DETAILED COMPARISON
# ============================================================

print()
print()
print("============================================================")
print("                 DETAILED COMPARISON")
print("============================================================")

print()

print(
    f"{'Classifier':<18}"
    f"{'Base Acc':<10}"
    f"{'Prop Acc':<10}"
    f"{'Base F1':<10}"
    f"{'Prop F1':<10}"
    f"{'Base AUC':<10}"
    f"{'Prop AUC':<10}"
)

print("-" * 78)

for name in baseline_results:

    base = baseline_results[name]
    prop = proposed_results[name]

    print(
        f"{name:<18}"
        f"{base['accuracy']:<10.4f}"
        f"{prop['accuracy']:<10.4f}"
        f"{base['f1']:<10.4f}"
        f"{prop['f1']:<10.4f}"
        f"{base['roc_auc']:<10.4f}"
        f"{prop['roc_auc']:<10.4f}"
    )


# ============================================================
# 10. ERROR RATE COMPARISON
# ============================================================

print()
print()
print("============================================================")
print("                 ERROR RATE COMPARISON")
print("============================================================")

print()

print(
    f"{'Classifier':<18}"
    f"{'Base FPR':<12}"
    f"{'Prop FPR':<12}"
    f"{'Base FNR':<12}"
    f"{'Prop FNR':<12}"
)

print("-" * 66)

for name in baseline_results:

    base = baseline_results[name]
    prop = proposed_results[name]

    print(
        f"{name:<18}"
        f"{base['fpr']:<12.4f}"
        f"{prop['fpr']:<12.4f}"
        f"{base['fnr']:<12.4f}"
        f"{prop['fnr']:<12.4f}"
    )


# ============================================================
# 11. TIME COMPARISON
# ============================================================

print()
print()
print("============================================================")
print("                    TIME COMPARISON")
print("============================================================")

print()

print(
    f"{'Classifier':<18}"
    f"{'Base Train':<14}"
    f"{'Prop Train':<14}"
    f"{'Base Detect':<14}"
    f"{'Prop Detect':<14}"
)

print("-" * 78)

for name in baseline_results:

    base = baseline_results[name]
    prop = proposed_results[name]

    print(
        f"{name:<18}"
        f"{base['training_time']:<14.4f}"
        f"{prop['training_time']:<14.4f}"
        f"{base['detection_time']:<14.4f}"
        f"{prop['detection_time']:<14.4f}"
    )


# ============================================================
# 12. FEATURE COUNT
# ============================================================

print()
print()
print("============================================================")
print("                    FEATURE COUNT")
print("============================================================")

print()

print("Baseline features :", X_baseline.shape[1])
print("Proposed features :", X_proposed.shape[1])
print(
    "Additional features in proposed method :",
    X_proposed.shape[1] - X_baseline.shape[1]
)


# ============================================================
# 13. COMPLETION
# ============================================================

print()
print()
print("============================================================")
print("             EXPERIMENT COMPLETED SUCCESSFULLY")
print("============================================================")