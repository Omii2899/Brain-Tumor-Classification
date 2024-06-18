import streamlit as st
import requests
from pathlib import Path
from streamlit.logger import get_logger
from PIL import Image
import base64
import io


FASTAPI_BACKEND_ENDPOINT = "http://localhost:8501"

# Make sure you have brain_tumor_model.pkl file in the FastAPI folder
FASTAPI_BRAIN_TUMOR_MODEL_LOCATION = Path(__file__).resolve().parents[2] / 'FastAPI_Labs' / 'src' / 'brain_tumor_model.pkl'

# Streamlit logger
LOGGER = get_logger(__name__)

# Streamlit App
def main():
# Set the main dashboard page browser tab title and icon
    st.set_page_config(
        page_title="Brain Tumor Classification",
        page_icon="🧠",
    )

    # Build the sidebar first
    with st.sidebar:
        # Check the status of backend
        try:
            backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT)
            if backend_request.status_code == 200:
                st.success("Backend online ✅")
            else:
                st.warning("Problem connecting 😭")
        except requests.ConnectionError as ce:
            LOGGER.error(ce)
            LOGGER.error("Backend offline 😱")
            st.error("Backend offline 😱")



    # Dashboard body
    image_path = "images/webapp_image.png"

    # Load the image
    image = Image.open(image_path)

    # Display the image with a specific width and centered
    st.image(image, use_column_width=True)

    st.write("# Brain Tumor Classification! 🧠")
    st.write("""
    ## Introduction
    This MLOps project aims to classify brain tumors using MRI images. 
    Please upload a brain MRI image to get started.
    """)


     # Image upload section
    uploaded_image = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])

   # Check if client has provided an input image file
    if uploaded_image:
        st.write('Preview Image')
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=False, width=300)
        st.session_state["IS_IMAGE_FILE_AVAILABLE"] = True
    else:
        st.session_state["IS_IMAGE_FILE_AVAILABLE"] = False

    # Predict button
    predict_button = st.button('Predict')

    if predict_button and uploaded_image:
        # Send the image to the FastAPI server for prediction
        files = {"file": uploaded_image.getvalue()}
        response = requests.post(f"{FASTAPI_BACKEND_ENDPOINT}/predict/", files=files)

        if response.status_code == 200:
            prediction = response.json().get("prediction")
            st.write(f"Prediction: {prediction}")
        else:
            st.write("Error: Could not get a prediction.")

if __name__ == "__main__":
    main()