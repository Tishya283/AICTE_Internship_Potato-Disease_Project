import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Function to make predictions
def model_prediction(test_image):
    # Load the trained model
    model = tf.keras.models.load_model("Train_potato_disease_model.keras")
    
    # Convert the image to the required format
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image) 
    input_arr = np.array([input_arr])
    
    # Make prediction
    predictions = model.predict(input_arr)
    
    # Return the index of the class with the highest probability
    return np.argmax(predictions) 

# Streamlit sidebar
st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Recognition'])

# Load local image
local_image_path = r"C:\Users\Tishya\Desktop\Potato_decies\download.jpg"

try:
    local_img = Image.open(local_image_path)
    st.image(local_img, caption="Local Image: Potato Disease", use_container_width=True)  # Updated parameter
except Exception as e:
    st.error(f"Error loading local image: {e}")

# Home page content
if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease System for Sustainable Agriculture By using Machine Learning</h1>", unsafe_allow_html=True)

# Recognition page content
elif app_mode == "Recognition":
    st.header("Plant Disease Recognition System")

    # File uploader for image input (optional)
    test_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        # Show the uploaded image
        st.image(test_image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

    # Prediction button
    if st.button("Predict"):
        if test_image is not None:
            st.snow()  # Optional snow effect during prediction
            st.write('Prediction in progress...')

            # Get prediction
            result_index = model_prediction(test_image)

            # Class names (adjust these if needed based on your model)
            class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___Healthy"]

            # Display the result
            st.success(f"The model predicts it as: {class_names[result_index]}")
        else:
            st.error("Please upload an image to make a prediction.")
