
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
dataset_path = BASE_DIR / "dataset"
results_path = BASE_DIR / "results"

IMG_SIZE = 224
BATCH_SIZE = 16

# Load dataset
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    str(dataset_path),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=42,
    shuffle=False
)

validation_data = datagen.flow_from_directory(
    str(dataset_path),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=42,
    shuffle=False
)

# Feature extraction using MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

print("\nExtracting Training Features...")
X_train = base_model.predict(train_data)
y_train = train_data.classes

print("\nExtracting Validation Features...")
X_val = base_model.predict(validation_data)
y_val = validation_data.classes

# Train SVM with probability estimation
print("\nTraining SVM Model...")

svm = SVC(
    kernel="linear",
    C=1.0,
    probability=True,
    random_state=42
)

svm.fit(X_train, y_train)

# Prediction
y_pred = svm.predict(X_val)

# Evaluation
print("\nSVM Evaluation Results")
print("----------------------")
print("Accuracy:", accuracy_score(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_val,
    y_pred,
    target_names=list(train_data.class_indices.keys()),
    zero_division=0
))

# Save model
os.makedirs(results_path, exist_ok=True)

joblib.dump(
    svm,
    results_path / "potato_svm_model.pkl"
)

print("\nSVM Model Training Completed!")
print("SVM Model Saved Successfully!")
