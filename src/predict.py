
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model(
    "../results/potato_disease_model.keras"
)

# Class names
class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# Image path
img_path = input("Enter image path: ")

# Load image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)
result = class_names[np.argmax(prediction)]

print("Prediction:", result)
