import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("best_model.keras")

# ✅ ALL 10 CLASSES
class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

st.title("Satellite Image Classifier (Federated Learning)")

st.write("⚠️ Upload satellite (top-view) images only")

uploaded_file = st.file_uploader(
    "Upload satellite image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # ✅ ensure RGB
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    # preprocess
    image = image.resize((64, 64))
    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0)

    # prediction
    prediction = model.predict(image)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
    top3 = np.argsort(prediction[0])[-3:][::-1]

    st.subheader("Top Predictions:")
    for i in top3:
        st.write(f"{class_names[i]}: {prediction[0][i]*100:.2f}%")