import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Multiclass Image Classification",
    layout="centered"
)

st.title("🐶🐱🐴 Multiclass Image Classification")
st.write("Upload an image and the model will predict the class.")

# ----------------------------
# Load SavedModel (Keras 3 export)
# ----------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "model", "saved_model")
    return tf.saved_model.load(model_path)

model = load_model()
infer = model.signatures["serving_default"]

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
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

# ----------------------------
# File uploader
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

        input_tensor = preprocess_image(image)

        # ✅ Correct SavedModel inference
        outputs = infer(input_tensor)
        predictions = list(outputs.values())[0].numpy()[0]

        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[predicted_index] * 100

        st.subheader("🔍 Prediction Result")
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.subheader("📊 Class Probabilities")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {predictions[i]*100:.2f}%")

    except Exception as e:
        st.error("❌ Error processing image")
        st.exception(e)

st.markdown("---")
st.markdown(
    "Developed as an **AI/ML Project** using TensorFlow SavedModel, Keras 3, and Streamlit 🚀"
)
