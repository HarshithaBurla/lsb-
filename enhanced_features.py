import numpy as np


def extract_weight_statistics(weights):
    weights = np.asarray(weights, dtype=np.float32)

    features = [
        np.mean(weights),
        np.std(weights),
        np.var(weights),
        np.min(weights),
        np.max(weights),
        np.median(weights),
        np.mean(np.abs(weights))
    ]

    return np.array(features, dtype=np.float32)


def extract_lsb_features(weights, max_lsb=5):
    weights = np.asarray(weights, dtype=np.float32)
    uint_weights = weights.view(np.uint32)

    features = []

    # Analyze each of the last 5 individual bit positions
    for bit in range(max_lsb):

        # Extract one bit: 0 or 1
        bit_values = (uint_weights >> bit) & 1
        bit_values = bit_values.astype(np.float32)

        # 1. Bit density
        # Percentage of weights having this bit set to 1
        bit_density = np.mean(bit_values)

        # 2. Bit entropy
        # Measures how random/balanced the bit values are
        p = bit_density

        if p == 0 or p == 1:
            entropy = 0.0
        else:
            entropy = -(p * np.log2(p) +
                        (1 - p) * np.log2(1 - p))

        # 3. Transition rate
        # Measures how frequently this bit changes
        if len(bit_values) > 1:
            transition_rate = np.mean(
                bit_values[1:] != bit_values[:-1]
            )
        else:
            transition_rate = 0.0

        features.extend([
            bit_density,
            entropy,
            transition_rate
        ])

    return np.array(features, dtype=np.float32)
def extract_enhanced_features(weights):
    weight_features = extract_weight_statistics(weights)
    lsb_features = extract_lsb_features(weights, max_lsb=5)

    return np.concatenate([
        weight_features,
        lsb_features
    ])


if __name__ == "__main__":
    print("Testing enhanced feature extraction...")

    sample_weights = np.array([
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        -0.10,
        -0.20,
        0.80
    ], dtype=np.float32)

    features = extract_enhanced_features(sample_weights)

    print()
    print("Enhanced feature extraction is working.")
    print("Number of features:", len(features))
    print("Features:")
    print(features)
