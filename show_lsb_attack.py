import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# ---------------------------------------------------------
# 1. Load original MNIST dataset
# ---------------------------------------------------------

mnist = datasets.MNIST(
    root=r".\data\mnist_original",
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

# ---------------------------------------------------------
# 2. Load original MNIST model weights
# ---------------------------------------------------------

original_weights = np.load(
    r".\data\model_zoos\mnist\zoo_weights.npy"
)

# ---------------------------------------------------------
# 3. Load 1-LSB attacked model weights
# ---------------------------------------------------------

attacked_file = np.load(
    r".\data\embedded\zoo_mnist_malwarename_malware_12584bytes_fill_True_msb_True.npz"
)

attacked_weights = attacked_file["1"]

print("Original model weights:", original_weights.shape)
print("1-LSB attacked weights:", attacked_weights.shape)

# Use the first model
original_model = original_weights[0]
attacked_model = attacked_weights[0]

# ---------------------------------------------------------
# 4. Create figure
# ---------------------------------------------------------

fig = plt.figure(figsize=(14, 12))

# ---------------------------------------------------------
# TOP: Original MNIST samples
# ---------------------------------------------------------

ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)

# Show 10 MNIST samples
for i in range(10):
    image, label = mnist[i]

    # Create small positions
    x = (i % 5) * 0.2
    y = 0.5 if i < 5 else 0

    # Add image
    ax1.imshow(
        image.squeeze(),
        cmap="gray",
        extent=(x, x + 0.15, y, y + 0.35)
    )

    ax1.text(
        x + 0.075,
        y - 0.05,
        str(label),
        ha="center",
        fontsize=10
    )

ax1.set_xlim(0, 1)
ax1.set_ylim(-0.15, 0.95)
ax1.axis("off")
ax1.set_title(
    "Original MNIST Dataset - Handwritten Digit Samples",
    fontsize=15,
    fontweight="bold"
)

# ---------------------------------------------------------
# BOTTOM LEFT: Original model
# ---------------------------------------------------------

ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2)

ax2.imshow(
    original_model.reshape(1, -1),
    aspect="auto",
    cmap="gray"
)

ax2.set_title(
    "Original MNIST Model Weights\n(BENIGN)",
    fontsize=14,
    fontweight="bold"
)

ax2.set_xlabel("Model weight parameters")
ax2.set_ylabel("Model")

# ---------------------------------------------------------
# BOTTOM RIGHT: 1-LSB attacked model
# ---------------------------------------------------------

ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2)

ax3.imshow(
    attacked_model.reshape(1, -1),
    aspect="auto",
    cmap="gray"
)

ax3.set_title(
    "1-LSB Modified MNIST Model Weights\n(MALICIOUS / ATTACKED)",
    fontsize=14,
    fontweight="bold"
)

ax3.set_xlabel("Model weight parameters")
ax3.set_ylabel("Model")

plt.suptitle(
    "MNIST Model and LSB-Based Model-Weight Attack",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

output = r".\data\mnist_original\mnist_lsb_attack_visualization.png"

plt.savefig(
    output,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print()
print("Visualization saved:")
print(output)