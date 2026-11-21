
import os
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = "../dataset"

classes = os.listdir(dataset_path)

class_counts = {}
image_sizes = []

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    count = 0

    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)

        try:
            with Image.open(file_path) as img:
                image_sizes.append(img.size)
                count += 1
        except:
            continue

    class_counts[class_name] = count

print("DATASET ANALYSIS")
print("----------------------")

total = 0

for name, count in class_counts.items():
    print(name, ":", count, "images")
    total += count

print("----------------------")
print("Total Images:", total)

# Class distribution graph
plt.figure(figsize=(9, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.title("Potato Disease Dataset Distribution")
plt.xlabel("Disease Classes")
plt.ylabel("Number of Images")
plt.xticks(rotation=15)
plt.tight_layout()

os.makedirs("../results", exist_ok=True)
plt.savefig("../results/class_distribution.png")
plt.show()

# Image dimension analysis
if image_sizes:
    print("Sample Image Dimensions:", image_sizes[0])
    print("Total Images Analyzed:", len(image_sizes))

from collections import Counter

dimension_counts = Counter()
mode_counts = Counter()

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)

        try:
            with Image.open(file_path) as img:
                dimension_counts[img.size] += 1
                mode_counts[img.mode] += 1

        except:
            continue

print("\nIMAGE PREPROCESSING ANALYSIS")
print("-----------------------------")

print("Image Dimensions:")
for dimension, count in dimension_counts.items():
    print(dimension, ":", count)

print("\nImage Color Modes:")
for mode, count in mode_counts.items():
    print(mode, ":", count)

# Test resizing and normalization
sample_path = os.path.join(
    dataset_path,
    "Potato___healthy",
    os.listdir(os.path.join(dataset_path, "Potato___healthy"))[0]
)

with Image.open(sample_path) as img:
    img = img.convert("RGB")
    img = img.resize((224, 224))

    import numpy as np
    img_array = np.array(img) / 255.0

    print("\nAfter Preprocessing:")
    print("Image Shape:", img_array.shape)
    print("Pixel Minimum:", img_array.min())
    print("Pixel Maximum:", img_array.max())
