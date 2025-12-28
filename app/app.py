import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="Multiclass Image Classification",
    layout="centered"
)

st.title("Multiclass Image Classification")
st.write("Upload an image of a **cat**, **dog**, or **horse**")

# -------------------------------
# Load Model (Inference Only)
# -------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),  # app/
        "..",                       # project root
        "model",
        "image_classifier_deploy.keras"
    )
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

# -------------------------------
# Class Labels (MUST match training order)
# -------------------------------
class_names = ["cats", "dogs", "horses"]

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Load & display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # -------------------------------
        # Preprocessing (MUST match training)
        # -------------------------------
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # -------------------------------
        # Prediction
        # -------------------------------
        predictions = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # -------------------------------
        # Results
        # -------------------------------
        st.subheader("Prediction Result")
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.subheader("Class Probabilities")
        for cls, prob in zip(class_names, predictions):
            st.write(f"{cls}: {prob * 100:.2f}%")

    except Exception as e:
        st.error("Error processing the image.")
        st.write(str(e))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built using TensorFlow, Keras & Streamlit")
