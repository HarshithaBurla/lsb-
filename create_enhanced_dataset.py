import numpy as np
from data_locator import ret_zoo_weights
from enhanced_features import extract_enhanced_features


print("Loading MNIST model weights...")

weights = ret_zoo_weights("mnist")

print("Weights shape:", weights.shape)

all_features = []

print("Extracting enhanced features...")

for i in range(len(weights)):

    features = extract_enhanced_features(weights[i])

    all_features.append(features)

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1} / {len(weights)} models")


all_features = np.array(all_features)

print("Final feature dataset shape:", all_features.shape)

output_path = "data/features/enhanced_mnist_benign.npy"

np.save(output_path, all_features)

print("Enhanced benign feature dataset saved successfully.")
print("File:", output_path)