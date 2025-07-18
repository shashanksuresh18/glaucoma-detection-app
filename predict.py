# predict.py (Corrected Logic)

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- 1. Load your saved glaucoma detection model ---
try:
    # Suppress the warning about the model compilation
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    model = tf.keras.models.load_model('glaucoma_detector_model.h5', compile=False)
    # Re-compile the model to have the metrics available (optional but good practice)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    print("Model loaded successfully. ✅")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure 'glaucoma_detector_model.h5' is in the same folder.")
    exit()

# --- 2. Set up the path to the image you want to test ---
# You can change this filename to test any image in your data/Images folder
image_filename = '4_0.jpg' # Example: An image that is truly GON+
image_path = os.path.join('data', 'Images', image_filename)

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: The image file was not found at '{image_path}'")
    exit()

# Define the image size the model expects
IMG_SIZE = (299, 299)

# --- 3. Load and preprocess the image ---
img = image.load_img(image_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0
print(f"Image '{image_filename}' preprocessed successfully. ✅")

# --- 4. Make a prediction ---
prediction = model.predict(img_array)
confidence = prediction[0][0]

# --- 5. Interpret and display the result (LOGIC IS NOW REVERSED) ---
print("\n--- Prediction Result ---")

# A HIGH score (closer to 1.0) now means NORMAL
if confidence > 0.5:
    print(f"Result: Normal (Confidence: {confidence*100:.2f}%)")
# A LOW score (closer to 0.0) now means GLAUCOMA
else:
    # We calculate (1 - confidence) to show a more intuitive "Confidence for Glaucoma"
    print(f"Result: Glaucoma Detected (Confidence: {(1-confidence)*100:.2f}%)")