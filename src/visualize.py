
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = "../dataset"

classes = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight"
]

random.seed(42)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for row, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, class_name)

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    selected_images = random.sample(images, min(3, len(images)))

    for col, image_name in enumerate(selected_images):
        image_path = os.path.join(class_path, image_name)

        img = Image.open(image_path)

        axes[row, col].imshow(img)
        axes[row, col].set_title(class_name.replace("Potato___", ""))
        axes[row, col].axis("off")

plt.tight_layout()

os.makedirs("../results", exist_ok=True)
plt.savefig("../results/sample_images.png")

print("Sample image visualization completed!")
print("Graph saved in results/sample_images.png")
