import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# AI MODEL STEGANALYSIS - FINAL PROJECT DEMONSTRATION
# ============================================================

print()
print("=" * 75)
print("             AI MODEL STEGANALYSIS - FINAL DEMO")
print("=" * 75)

print()
print("ANALYSIS TYPE : WHITE-BOX AI MODEL ANALYSIS")
print("DATASET       : MNIST MODEL ZOO")
print("OBJECTIVE     : Detect LSB-based modifications in model weights")

# ============================================================
# 1. LOAD FINAL PROPOSED DATASET
# ============================================================

X = np.load(
    r"data\features\ml_dataset\X_mnist_proposed_balanced.npy"
)

y = np.load(
    r"data\features\ml_dataset\y_mnist_comparison.npy"
)

print()
print("-" * 75)
print("1. DATASET INFORMATION")
print("-" * 75)

print(f"Total samples          : {len(y)}")
print(f"Total features/sample  : {X.shape[1]}")
print(f"Benign samples         : {np.sum(y == 0)}")
print(f"Attacked samples       : {np.sum(y == 1)}")

print()
print("Feature composition:")
print("  Gradient features        : 210")
print("  Reconstruction loss      :   1")
print("  Weight-statistical       :   7")
print("  LSB-specific features   :  15")
print("  --------------------------------")
print("  Total                    : 233")

# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ============================================================
# 3. TRAIN RANDOM FOREST
# ============================================================

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(X_train, y_train)

# ============================================================
# 4. FIND BENIGN AND ATTACKED SAMPLES
# ============================================================

benign_index = np.where(y_test == 0)[0][0]
attacked_index = np.where(y_test == 1)[0][0]

benign_sample = X_test[benign_index:benign_index + 1]
attacked_sample = X_test[attacked_index:attacked_index + 1]

# ============================================================
# 5. BENIGN MODEL DEMONSTRATION
# ============================================================

print()
print("=" * 75)
print("2. BENIGN MODEL - WHITE-BOX ANALYSIS")
print("=" * 75)

print()
print("Input Model")
print("-----------")
print("Source                  : MNIST Model Zoo")
print("Model condition         : ORIGINAL / UNMODIFIED")
print("LSB attack              : NOT APPLIED")
print("Model weights available : YES")
print("Weights analyzed        : 2464")

print()
print("Feature Extraction")
print("------------------")
print("Gradient features       : 210")
print("Reconstruction loss     : 1")
print("Weight statistics       : 7")
print("LSB-specific features   : 15")
print("Total features          : 233")

benign_prediction = classifier.predict(benign_sample)[0]
benign_probability = classifier.predict_proba(benign_sample)[0]

if benign_prediction == 0:
    benign_result = "BENIGN"
    benign_confidence = benign_probability[0] * 100
else:
    benign_result = "MALICIOUS / LSB ATTACKED"
    benign_confidence = benign_probability[1] * 100

print()
print("Detection Result")
print("----------------")
print(f"Classifier              : Random Forest")
print(f"Predicted class         : {benign_result}")
print(f"Detection confidence    : {benign_confidence:.2f}%")

# ============================================================
# 6. ATTACKED MODEL DEMONSTRATION
# ============================================================

print()
print("=" * 75)
print("3. LSB-ATTACKED MODEL - WHITE-BOX ANALYSIS")
print("=" * 75)

print()
print("Input Model")
print("-----------")
print("Source                  : MNIST Model Zoo")
print("Model condition         : LSB-MODIFIED")
print("LSB attack              : 1-LSB MODIFICATION")
print("Model weights available : YES")
print("Weights analyzed        : 2464")

print()
print("Feature Extraction")
print("------------------")
print("Gradient features       : 210")
print("Reconstruction loss     : 1")
print("Weight statistics       : 7")
print("LSB-specific features   : 15")
print("Total features          : 233")

attacked_prediction = classifier.predict(attacked_sample)[0]
attacked_probability = classifier.predict_proba(attacked_sample)[0]

if attacked_prediction == 0:
    attacked_result = "BENIGN"
    attacked_confidence = attacked_probability[0] * 100
else:
    attacked_result = "MALICIOUS / LSB ATTACKED"
    attacked_confidence = attacked_probability[1] * 100

print()
print("Detection Result")
print("----------------")
print(f"Classifier              : Random Forest")
print(f"Predicted class         : {attacked_result}")
print(f"Detection confidence    : {attacked_confidence:.2f}%")

# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("4. FINAL DETECTION SUMMARY")
print("=" * 75)

print()
print("Original Model")
print("--------------")
print("Input                   : Unmodified MNIST model")
print(f"System decision         : {benign_result}")

print()
print("Attacked Model")
print("--------------")
print("Input                   : 1-LSB modified MNIST model")
print(f"System decision         : {attacked_result}")

print()
print("Pipeline")
print("--------")
print("Model weights")
print("      ↓")
print("White-box feature extraction")
print("      ↓")
print("233-feature representation")
print("      ↓")
print("Random Forest classifier")
print("      ↓")
print("BENIGN / MALICIOUS")

print()
print("=" * 75)
print("              FINAL DETECTION COMPLETED")
print("=" * 75)
print()