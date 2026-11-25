from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

root = Path(r"E:\Enhanced_AI_Model_Steganalysis")

extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
images = [
    p for p in root.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
    and "venv" not in p.parts
]

print("Image files found:", len(images))

if images:
    for i, path in enumerate(images[:10], start=1):
        print(f"{i}. {path}")

    image_path = images[0]
    print("\nDisplaying actual image:", image_path)

    with Image.open(image_path) as img:
        print("Image size:", img.size)
        print("Image mode:", img.mode)
        plt.imshow(img.copy(), cmap="gray" if img.mode == "L" else None)
        plt.title(image_path.name)
        plt.axis("off")
        plt.show()
else:
    print("No individual image files found.")
    print("Search for .npy, .npz, .pt, or .pth dataset files next.")
