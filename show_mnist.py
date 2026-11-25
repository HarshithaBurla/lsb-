from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

mnist = datasets.MNIST(
    root=r".\data\mnist_original",
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

print("MNIST DATASET")
print("Total training images:", len(mnist))
print("Image size:", mnist[0][0].shape)

# 30 images = 5 columns x 6 rows
fig, axes = plt.subplots(6, 5, figsize=(12, 14))

for i, ax in enumerate(axes.flat):
    image, label = mnist[i]

    ax.imshow(image.squeeze().numpy(), cmap="gray", interpolation="nearest")
    ax.set_title(f"Sample {i+1} | Digit {label}", fontsize=11)
    ax.axis("off")

plt.suptitle(
    "Original MNIST Dataset - 30 Handwritten Digit Samples",
    fontsize=18,
    fontweight="bold"
)

plt.subplots_adjust(
    top=0.94,
    hspace=0.35,
    wspace=0.25
)

output_path = r".\data\mnist_original\mnist_30_samples.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print()
print("30 MNIST samples saved successfully:")
print(output_path)