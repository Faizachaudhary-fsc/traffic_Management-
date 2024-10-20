import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model('traffic_management_cnn_model')
    return model

model = load_trained_model()

# Define class names
class_names = ['Car', 'Motorcycle', 'Truck']

# Streamlit App
st.title('Traffic Management System')

st.write("""
         Upload an image of traffic, and the model will detect and classify the vehicles.
         """)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    img = image_pil.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    st.write(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")
