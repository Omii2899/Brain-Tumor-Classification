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

    def display_image_from_base64(image_base64):
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image)

    # def display_image_from_bytes(image_bytes):
    #     image = Image.open(io.BytesIO(image_bytes))
    #     st.image(image)

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

    # if predict_button and uploaded_image:
    #     # Send the image to the FastAPI server for prediction
    #     files = {"file": uploaded_image.getvalue()}
    #     #response = requests.post(f"{FASTAPI_BACKEND_ENDPOINT}/predict/", files=files)
    #     response = requests.post(f"{FASTAPI_BACKEND_ENDPOINT}/predict/", files=files)

    #     if response.status_code == 200:
    #         prediction = response.json().get("prediction")
    #         st.write(f"Prediction: {prediction}")
    #     else:
    #         st.write("Error: Could not get a prediction.")
    if predict_button and uploaded_image:
        # Convert image to JPEG format in memory
        image_buffer = io.BytesIO()
        image.save(image_buffer, format='JPEG')
        image_buffer.seek(0)

        # Send the image to the FastAPI server for prediction
        files = {"file": ("image.jpg", image_buffer, "image/jpeg")}
        response = requests.post(f"{FASTAPI_BACKEND_ENDPOINT}/predict/", files=files)

        if response.status_code == 200:
            # result = response.json()
            # prediction = result['Prediction']

            # # image = Image.open(io.BytesIO(response.content))
            # # st.image(image, caption='Image from FastAPI', use_column_width=False, width=300)
            # inference = result['Inference']
            # boundary = result['Boundaries']

            # st.write(f"Prediction: {prediction}")
            # st.write("Inference:") 
            # display_image_from_bytes(inference) 
            # st.write("Boundary Image:") 
            # display_image_from_bytes(boundary)
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
            # st.image(inference_image, caption='Explanation', use_column_width=False, width=300)
            # st.image(boundaries_image, caption='Marked Boundaries', use_column_width=False, width=300)
            # inference_image = result['Inference']
            # boundaries_image = result['Boundaries']

            # st.write(f"Prediction: {prediction}")
            # st.write("Inference:")
            # st.image(inference_image, caption='Inference Image', use_column_width=False, width=300)
            # st.write("Boundary Image:")
            # st.image(boundaries_image, caption='Boundary Image', use_column_width=False, width=300)
            #--
            # result = response.json()
            # prediction = result['Prediction']
            # inference_base64 = result['Inference']
            # boundaries_base64 = result['Boundaries']

            # st.write(f"Prediction: {prediction}")
            # st.write("Inference:") 
            # display_image_from_base64(inference_base64)
            # st.write("Boundary Image:") 
            # display_image_from_base64(boundaries_base64)
            #--
        else:
            st.write("Error: Could not get a prediction.")

if __name__ == "__main__":
    main()
