import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "model",
        "image_classifier.keras"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

class_names = ["cats", "dogs", "horses"]

st.title("Multiclass Image Classification")
st.write("Upload an image of a cat, dog, or horse")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.subheader("Prediction")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Probabilities")
    for cls, prob in zip(class_names, predictions):
        st.write(f"{cls}: {prob*100:.2f}%")
