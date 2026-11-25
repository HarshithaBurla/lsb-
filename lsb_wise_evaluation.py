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
    roc_auc_score
)

import xgboost as xgb


# ============================================================
# SETTINGS
# ============================================================

BENIGN_FILE = "data/features/enhanced_mnist_benign.npy"

BASELINE_GRADS = (
    "data/features/grads/"
    "zoo_mnist_malwarename_malware_12584bytes_"
    "fill_True_msb_True.npz"
)

LOSS_FILE = (
    "data/features/losses/"
    "zoo_mnist_malwarename_malware_12584bytes_"
    "fill_True_msb_True.npz"
)

ATTACKED_PREFIX = "data/features/enhanced_mnist_attacked_"

SEED = 42

N_BENIGN = 100
N_ATTACKED = 100


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LSB-WISE BASELINE vs PROPOSED EVALUATION")
print("=" * 70)

print()
print("Loading benign enhanced features...")

benign_enhanced = np.load(BENIGN_FILE)

print("Benign enhanced shape:", benign_enhanced.shape)


print()
print("Loading gradient features...")

gradient_data = np.load(BASELINE_GRADS)

print("Gradient data loaded.")


print()
print("Loading reconstruction loss...")

loss_data = np.load(LOSS_FILE)

print("Loss data loaded.")


# ============================================================
# PREPARE BASELINE BENIGN FEATURES
# ============================================================

benign_gradients = gradient_data["0"][:N_BENIGN]

benign_losses = loss_data["losses"][0, :N_BENIGN]

benign_losses = benign_losses.reshape(-1, 1)

baseline_benign = np.concatenate(
    [benign_gradients, benign_losses],
    axis=1
)

proposed_benign = np.concatenate(
    [baseline_benign, benign_enhanced[:N_BENIGN]],
    axis=1
)

print()
print("Baseline benign shape :", baseline_benign.shape)
print("Proposed benign shape :", proposed_benign.shape)


# ============================================================
# CLASSIFIERS
# ============================================================

def create_classifiers():

    return {

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=SEED,
            n_jobs=-1
        ),

        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=-1
        ),

        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=300,
                early_stopping=True,
                random_state=SEED
            ))
        ])
    }


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, X_train, X_test, y_train, y_test):

    start = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start

    start = time.time()

    predictions = model.predict(X_test)

    detect_time = time.time() - start

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)[:, 1]

    else:

        probabilities = predictions

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

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "train_time": train_time,
        "detect_time": detect_time
    }


# ============================================================
# STORE RESULTS
# ============================================================

results = {
    "Random Forest": [],
    "XGBoost": [],
    "MLP": []
}


# ============================================================
# LSB 1 TO 23
# ============================================================

for lsb in range(1, 24):

    print()
    print("=" * 70)
    print(f"PROCESSING {lsb}-LSB ATTACK")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD ATTACKED ENHANCED FEATURES
    # --------------------------------------------------------

    attacked_file = (
        f"{ATTACKED_PREFIX}{lsb}lsb.npy"
    )

    attacked_enhanced = np.load(
        attacked_file
    )

    attacked_enhanced = attacked_enhanced[
        :N_ATTACKED
    ]

    # --------------------------------------------------------
    # BASELINE ATTACKED FEATURES
    # --------------------------------------------------------

    attacked_gradients = gradient_data[
        str(lsb)
    ][:N_ATTACKED]

    attacked_losses = loss_data[
        "losses"
    ][lsb, :N_ATTACKED]

    attacked_losses = attacked_losses.reshape(
        -1,
        1
    )

    baseline_attacked = np.concatenate(
        [
            attacked_gradients,
            attacked_losses
        ],
        axis=1
    )

    # --------------------------------------------------------
    # PROPOSED ATTACKED FEATURES
    # --------------------------------------------------------

    proposed_attacked = np.concatenate(
        [
            baseline_attacked,
            attacked_enhanced
        ],
        axis=1
    )

    # --------------------------------------------------------
    # CREATE LABELS
    # --------------------------------------------------------

    y_benign = np.zeros(
        N_BENIGN,
        dtype=int
    )

    y_attack = np.ones(
        N_ATTACKED,
        dtype=int
    )

    y = np.concatenate(
        [
            y_benign,
            y_attack
        ]
    )

    # --------------------------------------------------------
    # COMBINE DATA
    # --------------------------------------------------------

    X_baseline = np.concatenate(
        [
            baseline_benign,
            baseline_attacked
        ],
        axis=0
    )

    X_proposed = np.concatenate(
        [
            proposed_benign,
            proposed_attacked
        ],
        axis=0
    )

    # --------------------------------------------------------
    # SAME TRAIN / TEST SPLIT
    # --------------------------------------------------------

    indices = np.arange(
        len(y)
    )

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=SEED
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    # --------------------------------------------------------
    # CLASSIFIERS
    # --------------------------------------------------------

    classifiers = create_classifiers()

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_results = {}

    for name, model in classifiers.items():

        result = evaluate(
            model,
            X_baseline[train_idx],
            X_baseline[test_idx],
            y_train,
            y_test
        )

        baseline_results[name] = result

    # --------------------------------------------------------
    # PROPOSED
    # --------------------------------------------------------

    proposed_results = {}

    classifiers = create_classifiers()

    for name, model in classifiers.items():

        result = evaluate(
            model,
            X_proposed[train_idx],
            X_proposed[test_idx],
            y_train,
            y_test
        )

        proposed_results[name] = result

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    for name in results:

        results[name].append({

            "lsb": lsb,

            "baseline": baseline_results[name],

            "proposed": proposed_results[name]

        })

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print(
        f"{'Classifier':16s}"
        f"{'Base Acc':>12s}"
        f"{'Prop Acc':>12s}"
        f"{'Base F1':>12s}"
        f"{'Prop F1':>12s}"
    )

    print("-" * 64)

    for name in results:

        b = baseline_results[name]
        p = proposed_results[name]

        print(
            f"{name:16s}"
            f"{b['accuracy']:12.4f}"
            f"{p['accuracy']:12.4f}"
            f"{b['f1']:12.4f}"
            f"{p['f1']:12.4f}"
        )


# ============================================================
# FINAL TABLES
# ============================================================

for classifier in [
    "Random Forest",
    "XGBoost",
    "MLP"
]:

    print()
    print()
    print("=" * 70)
    print(f"{classifier.upper()} — LSB-WISE RESULTS")
    print("=" * 70)

    print()

    print(
        f"{'LSB':>5s}"
        f"{'Base Acc':>12s}"
        f"{'Prop Acc':>12s}"
        f"{'Base F1':>12s}"
        f"{'Prop F1':>12s}"
        f"{'Base AUC':>12s}"
        f"{'Prop AUC':>12s}"
    )

    print("-" * 77)

    for item in results[classifier]:

        b = item["baseline"]
        p = item["proposed"]

        print(
            f"{item['lsb']:5d}"
            f"{b['accuracy']:12.4f}"
            f"{p['accuracy']:12.4f}"
            f"{b['f1']:12.4f}"
            f"{p['f1']:12.4f}"
            f"{b['auc']:12.4f}"
            f"{p['auc']:12.4f}"
        )


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 70)
print("LSB-WISE EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)