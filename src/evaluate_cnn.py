
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = "../dataset"

IMG_SIZE = 224
BATCH_SIZE = 32

# Validation dataset
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

validation_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=42,
    shuffle=False
)

# Load CNN model
model = tf.keras.models.load_model(
    "../results/potato_disease_model.keras"
)

# Predictions
predictions = model.predict(validation_data)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_data.classes

# Evaluation
print("\nCNN Evaluation Results")
print("----------------------")
print("Accuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(validation_data.class_indices.keys())
))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=list(validation_data.class_indices.keys()),
    yticklabels=list(validation_data.class_indices.keys())
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN Confusion Matrix")
plt.tight_layout()

os.makedirs("../results", exist_ok=True)
plt.savefig("../results/cnn_confusion_matrix.png")

print("\nCNN Confusion Matrix saved successfully!")
