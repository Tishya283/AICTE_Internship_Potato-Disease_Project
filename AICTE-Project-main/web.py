import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def model_prediction(test_image):
    model = tf.keras.models.load_model("Train_potato_disease_model.keras")
   
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image) 
    input_arr = np.array([input_arr])
  
    predictions = model.predict(input_arr)
  
    return np.argmax(predictions) 

st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Recognition'])

local_image_path = r"C:\Users\Ekta Vora\Desktop\Potato_decies\Screenshot 2025-02-06 185528.png"

try:
    local_img = Image.open(local_image_path)
    st.image(local_img, caption="Local Image: Potato Disease", use_container_width=True)
except Exception as e:
    st.error(f"Error loading local image: {e}")

if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif app_mode == "Recognition":
    st.header("Plant Disease Recognition System")

    test_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

    if st.button("Predict"):
        if test_image is not None:
            st.snow() 
            st.write('Prediction in progress...')

            result_index = model_prediction(test_image)

            class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___Healthy"]

            # Display the result
            st.success(f"The model predicts it as: {class_names[result_index]}")
        else:
            st.error("Please upload an image to make a prediction.")
