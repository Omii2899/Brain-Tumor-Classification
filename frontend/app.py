import streamlit as st
import requests
from pathlib import Path
from streamlit.logger import get_logger
from PIL import Image
import io
import base64


FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"

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

    st.write("# Brain Tumor Classification! 🧠")
    st.write("""
    ## Introduction
    This MLOps project aims to classify brain tumors using MRI images. 
    Please upload a brain MRI image to get started.
    """)

    # Image upload section
    uploaded_image = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg"])

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
        # Convert image to JPEG format in memory
        image_buffer = io.BytesIO()
        image.save(image_buffer, format='JPEG')
        image_buffer.seek(0)

        # Send the image to the FastAPI server for prediction
        files = {"file": ("image.jpg", image_buffer, "image/jpeg")}
        response = requests.post(f"{FASTAPI_BACKEND_ENDPOINT}/predict/", files=files)

        if response.status_code == 200:
            result = response.json()
            prediction = result['Prediction']

            # Decode the base64 images
            inference_image = Image.open(io.BytesIO(base64.b64decode(result['Inference'])))
            boundaries_image = Image.open(io.BytesIO(base64.b64decode(result['Boundaries'])))
            st.write(f"Prediction: {prediction}")

            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(inference_image, caption='Explanation', use_column_width=False, width=300)
            with col2:
                st.image(boundaries_image, caption='Marked Boundaries', use_column_width=False, width=300)

        elif response.status_code == 400:
            result = response.json()
            st.error(result['error'])
            # st.write('Preview Image')
            # st.image(image, caption='Uploaded Image', use_column_width=False, width=300)
            retry_button = st.button('Retry')
            if retry_button:
                st.session_state["IS_IMAGE_FILE_AVAILABLE"] = False  # Reset image availability flag
                st.experimental_rerun()  # Rerun the app to allow re-uploading
            #retry_button = st.button('Retry')
        elif response.status_code == 500:    
            st.write("Error: We dont understand the image")
        else:
            st.write(f"Error: {response.status_code}")

    if not uploaded_image or (uploaded_image and not st.session_state["IS_IMAGE_FILE_AVAILABLE"]):
        st.info("Please upload an image to proceed.")
if __name__ == "__main__":
    main()
