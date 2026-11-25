import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("Loading dataset...")

X = np.load(
    "data/features/ml_dataset/X_mnist_balanced.npy"
)

y = np.load(
    "data/features/ml_dataset/y_mnist_balanced.npy"
)

print("X shape:", X.shape)
print("y shape:", y.shape)


print()
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


print()
print("Standardizing features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Standardization completed.")


print()
print("Training MLP...")

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15
)

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("Training completed.")
print("Training time:", round(training_time, 2), "seconds")
print("Iterations:", model.n_iter_)


print()
print("Testing model...")

start_time = time.time()

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

detection_time = time.time() - start_time


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)
fnr = fn / (fn + tp)


print()
print("==============================")
print("MLP RESULTS")
print("==============================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

print()
print("Confusion Matrix:")
print(cm)

print()
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

print()
print("FPR:", round(fpr, 4))
print("FNR:", round(fnr, 4))

print()
print("Detection time:", round(detection_time, 6), "seconds")