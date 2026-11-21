
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "results" / "potato_svm_model.pkl"

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

display_names = {
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight": "Late Blight",
    "Potato___healthy": "Healthy"
}

st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Potato Leaf Disease Detection")
st.write("Upload a potato leaf image to detect its disease.")

@st.cache_resource
def load_models():
    svm = joblib.load(MODEL_PATH)

    feature_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )

    return svm, feature_model

try:
    svm, feature_model = load_models()

    uploaded_file = st.file_uploader(
        "Upload Potato Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

        if st.button("Predict Disease", type="primary"):
            with st.spinner("Analyzing image..."):

                resized_image = image.resize((224, 224))
                image_array = np.array(resized_image, dtype=np.float32)
                image_array = np.expand_dims(image_array, axis=0)
                image_array = preprocess_input(image_array)

                features = feature_model.predict(
                    image_array, verbose=0
                )

                prediction = svm.predict(features)[0]
                probabilities = svm.predict_proba(features)[0]

                result = class_names[prediction]
                confidence = float(probabilities[prediction]) * 100

            st.success("Prediction Completed!")

            if result == "Potato___healthy":
                st.success("Result: Potato Leaf is Healthy 🌿")
            elif result == "Potato___Early_blight":
                st.warning("Result: Early Blight Detected")
            else:
                st.error("Result: Late Blight Detected")

            st.metric(
                label="Estimated Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(int(round(confidence)))

            st.caption(
                "Note: This is an estimated model probability, "
                "not a guarantee of disease diagnosis."
            )

except Exception as e:
    st.error(f"Error: {e}")
