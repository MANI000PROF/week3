import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("wasteclassification_model.h5")

model = load_model()

# Define class labels (adjust based on your model's training labels)
class_labels = ['Organic Waste', 'Recyclable Waste']

# Streamlit UI
st.title("Waste Classification Using CNN")
st.write("Upload an image of waste, and the model will classify it as Organic Waste or Recyclable Waste.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = image.resize((224, 224))  # Adjust size as per your model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
