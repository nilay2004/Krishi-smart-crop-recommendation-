import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# --- Paths for assets and models (relative to project root) ---
CROP_IMAGE_PATH = "CROP-RECOMMENDATION/crop.png"
CROP_DATA_PATH = "CROP-RECOMMENDATION/Crop_recommendation.csv"
CROP_MODEL_PATH = "CROP-RECOMMENDATION/RF.pkl"

DISEASE_MODEL_PATH = "PLANT-DISEASE-IDENTIFICATION/trained_plant_disease_model.keras"
DISEASE_IMAGE_PATH = "PLANT-DISEASE-IDENTIFICATION/Diseases.png"

# --- Load Crop Recommendation Model and Data ---
# Check if the model file exists, if not, handle it (e.g., retrain or provide error)
if not os.path.exists(CROP_MODEL_PATH):
    st.error(f"Crop recommendation model not found at {CROP_MODEL_PATH}. Please ensure it exists.")
    # Placeholder for training or model loading logic if needed
    # For now, we'll assume it's pre-trained and available.
    RF_Model_pkl = None
else:
    try:
        RF_Model_pkl = pickle.load(open(CROP_MODEL_PATH, 'rb'))
    except Exception as e:
        st.error(f"Error loading crop recommendation model: {e}")
        RF_Model_pkl = None

# --- Crop Recommendation Function ---
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    if RF_Model_pkl is None:
        return "Model not loaded"
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction[0]

# --- Load Plant Disease Model ---
# Check if the model file exists (especially important for LFS managed files)
if not os.path.exists(DISEASE_MODEL_PATH):
    st.error(f"Plant disease model not found at {DISEASE_MODEL_PATH}. Please ensure it exists and Git LFS is correctly set up.")
    disease_model = None
else:
    try:
        disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading plant disease model: {e}")
        disease_model = None

# --- Plant Disease Prediction Function ---
def model_prediction(test_image):
    if disease_model is None:
        return -1 # Indicate model not loaded

    image = Image.open(test_image)
    image = image.resize((128, 128)) # Ensure correct size for the model
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch

    predictions = disease_model.predict(input_arr)
    return np.argmax(predictions)

# List of plant diseases (from original main.py)
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- Streamlit Application Layout ---
st.sidebar.title("KRISHI - Smart Agriculture Assistant")

# Sidebar image for overall app
if os.path.exists(CROP_IMAGE_PATH):
    try:
        img_crop = Image.open(CROP_IMAGE_PATH)
        st.sidebar.image(img_crop, caption="Crop Recommendation")
    except Exception as e:
        st.sidebar.error(f"Could not load crop image: {e}")

if os.path.exists(DISEASE_IMAGE_PATH):
    try:
        img_disease = Image.open(DISEASE_IMAGE_PATH)
        st.sidebar.image(img_disease, caption="Disease Recognition")
    except Exception as e:
        st.sidebar.error(f"Could not load disease image: {e}")

app_mode = st.sidebar.selectbox("Select Feature", ["Crop Recommendation", "Plant Disease Identification"])


# --- Crop Recommendation Section ---
if app_mode == "Crop Recommendation":
    st.markdown("<h1 style='text-align: center;'>SMART CROP RECOMMENDATIONS</h1>", unsafe_allow_html=True)

    st.header("Enter Crop Details")
    nitrogen = st.number_input("Nitrogen", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
    phosphorus = st.number_input("Phosphorus", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
    potassium = st.number_input("Potassium", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

    if st.button("Predict Crop"):
        inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        if np.isnan(inputs).any() or (inputs == 0).all():
            st.error("Please fill in all input fields with valid values before predicting.")
        else:
            predicted_crop = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            if predicted_crop == "Model not loaded":
                st.warning("Crop recommendation model could not be loaded. Please check logs for errors.")
            else:
                st.success(f"The recommended crop is: {predicted_crop.capitalize()}")

# --- Plant Disease Identification Section ---
elif app_mode == "Plant Disease Identification":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)
    st.header("Upload an Image for Disease Recognition")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict Disease"):
            if disease_model is None:
                st.warning("Plant disease model could not be loaded. Please check logs for errors.")
            else:
                st.snow()
                st.write("Our Prediction")
                result_index = model_prediction(test_image)

                if result_index != -1:
                    st.success(f"Model is predicting it's {class_name[result_index]}")
                else:
                    st.error("Could not make a prediction. Model might not be loaded.") 