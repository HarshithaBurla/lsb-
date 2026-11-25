import numpy as np

from data_locator import ret_embedded_weights
from enhanced_features import extract_enhanced_features


print("Loading MNIST attacked weights...")

embedded = ret_embedded_weights("mnist", msb=True)

print("Available LSB levels:", embedded.files)

for lsb in embedded.files:

    print(f"\nProcessing {lsb}-LSB attacked models...")

    weights = embedded[lsb]

    print("Weights shape:", weights.shape)

    all_features = []

    for i in range(len(weights)):

        features = extract_enhanced_features(weights[i])

        all_features.append(features)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} / {len(weights)} models")

    all_features = np.array(all_features)

    output_path = f"data/features/enhanced_mnist_attacked_{lsb}lsb.npy"

    np.save(output_path, all_features)

    print("Feature shape:", all_features.shape)
    print("Saved:", output_path)


print("\nAll attacked feature datasets created successfully.")