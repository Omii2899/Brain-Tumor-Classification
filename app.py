import streamlit as st
import requests
from pathlib import Path
from streamlit.logger import get_logger
from PIL import Image
import base64
import io


FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"

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

        st.info("Task Progress")
        task_progress = ["Image Received", "Preprocessing Image", "Running Model", "Generating Output", "Complete"]
        for task in task_progress:
            st.write(f"- {task}")


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

    # If predict button is pressed
    if predict_button:
        if "IS_IMAGE_FILE_AVAILABLE" in st.session_state and st.session_state["IS_IMAGE_FILE_AVAILABLE"]:
            if FASTAPI_BRAIN_TUMOR_MODEL_LOCATION.is_file():
                client_input = uploaded_image
                try:
                    result_container = st.empty()
                    with st.spinner('Predicting...'):
                        # Prepare the file for upload
                        files = {"file": client_input.getvalue()}
                        predict_response = requests.post(f'{FASTAPI_BACKEND_ENDPOINT}/predict', files=files)

                    if predict_response.status_code == 200:
                        prediction = predict_response.json()
                        if prediction["result"] == "No Tumor":
                            result_container.success("No Tumor Detected")
                        else:
                            result_container.error("Tumor Detected")
                    else:
                        st.toast(f':red[Status from server: {predict_response.status_code}. Refresh page and check backend status]', icon="🔴")
                except Exception as e:
                    st.toast(':red[Problem with backend. Refresh page and check backend status]', icon="🔴")
                    LOGGER.error(e)
            else:
                LOGGER.warning('brain_tumor_model.pkl not found in FastAPI Lab. Make sure to run train.py to get the model.')
                st.toast(':red[Model brain_tumor_model.pkl not found. Please run the train.py file in FastAPI Lab]', icon="🔥")
        else:
            LOGGER.error('Provide a valid MRI image file')
            st.toast(':red[Please upload a valid MRI image file]', icon="🛑")


if __name__ == "__main__":
    main()
# st.button("Start Model", on_click=)