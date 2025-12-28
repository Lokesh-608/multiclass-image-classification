import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Multiclass Image Classification",
    layout="centered"
)

st.title("🐶🐱🐴 Multiclass Image Classification")
st.write("Upload an image and the model will predict the class.")

# ----------------------------
# Load model (Keras 3 – exported model)
# ----------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "model", "saved_model")
    return tf.keras.models.load_model(model_path)

model = load_model()
st.success("✅ Model loaded successfully")

# ----------------------------
# Class names (same order as training)
# ----------------------------
class_names = ["cats", "dogs", "horses"]

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# Upload image
# ----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an image (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# Prediction
# ----------------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed_image = preprocess_image(image)

        # ✅ Keras 3 inference (NO .predict)
        predictions = model(processed_image, training=False).numpy()
        confidence_scores = predictions[0]

        predicted_index = np.argmax(confidence_scores)
        predicted_class = class_names[predicted_index]
        confidence = confidence_scores[predicted_index] * 100

        st.subheader("🔍 Prediction Result")
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.subheader("📊 Class Probabilities")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {confidence_scores[i]*100:.2f}%")

    except Exception as e:
        st.error("❌ Error processing image")
        st.exception(e)

st.markdown("---")
st.markdown(
    "Developed as an **AI/ML Project** using TensorFlow, Keras 3, and Streamlit 🚀"
)
