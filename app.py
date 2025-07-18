# app.py (Corrected for 4-Channel Images)

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# --- Constants ---
IMG_SIZE = (299, 299)
MODEL_PATH = 'glaucoma_detector_model.h5'

# --- Load the pre-trained model ---
@st.cache_resource
def load_my_model():
    """Loads and returns the trained Keras model."""
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Please make sure '{MODEL_PATH}' is in the correct directory.")
        return None

model = load_my_model()

# --- Helper Function to Preprocess Image ---
def preprocess_image(img):
    """Preprocesses the uploaded image to be model-ready."""
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# --- Streamlit App Interface ---
st.title("👁️ Glaucoma Detection App")
st.write(
    "This application uses a deep learning model to predict the presence of glaucoma "
    "from retinal fundus images. Upload an image to get a prediction."
)

uploaded_file = st.file_uploader("Choose an eye scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    with st.spinner('Classifying...'):
        try:
            pil_image = Image.open(uploaded_file)
            
            # --- THIS IS THE FIX ---
            # Convert the image to RGB (3 channels), removing any transparency channel
            pil_image = pil_image.convert('RGB')
            
            processed_image = preprocess_image(pil_image)
            prediction = model.predict(processed_image)
            confidence = prediction[0][0]

            st.write("---")
            st.subheader("Prediction Result")
            
            if confidence > 0.5:
                st.success(f"**Result: Normal** (Confidence: {confidence*100:.2f}%)")
            else:
                st.error(f"**Result: Glaucoma Detected** (Confidence: {(1-confidence)*100:.2f}%)")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")