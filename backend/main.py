from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.pyfunc
import os
import io
import base64
import numpy as np
from scripts.Model_Serve import Model_Server
from scripts.statistics_histogram import validate_image


app = FastAPI()

ms = Model_Server(stage='Staging')


@app.get("/")
def read_root():
    return {"message": "Welcome to the Brain Tumor Classification API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    # Read the uploaded image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Validate the image
    image_path = "/tmp/temp_image.jpg"
    image.save(image_path, format="JPEG")
    is_valid = validate_image(image_path)
    if not is_valid:
        return JSONResponse(content={"error": "Invalid image. Please upload a correct brain MRI image."}, status_code=400)
    
    # Get existing filenames in the cloud
    existing_filenames = ms.get_existing_filenames()

    # Generate a random file name for the image
    file_name = ms.generate_unique_filename(existing_filenames)

    # file_name = file.filename
    temp_file_path = f"/tmp/{file_name}"  # Temporary file path

    # image = Image.open(io.BytesIO(contents))
    image.save(temp_file_path, format="JPEG")

    #Make prediction
    prediction = ms.serve_model(image)
    pred_inf = ms.explain_pred()

    #Upload images to cloud bucked
    folder_name = f'InferenceLogs/ImageLogs/{prediction}/'
    ms.uploadtobucket(temp_file_path, file_name, folder_name)
    
    # Convert images to base64 strings to send as JSON response
    pil_inference = Image.fromarray((pred_inf[0] * 255).astype(np.uint8))
    pil_boundaries = Image.fromarray((pred_inf[1] * 255).astype(np.uint8))

    # Convert the PIL image to JPEG format
    pil_inference_buffer = io.BytesIO()
    pil_inference.save(pil_inference_buffer, format='JPEG')
    #pil_inference_buffer.seek(0)
    pil_boundaries_buffer = io.BytesIO()
    pil_boundaries.save(pil_boundaries_buffer, format='JPEG')
    #pil_boundaries_buffer.seek(0)

    # Encode the image bytes as base64 strings
    inference_base64 = base64.b64encode(pil_inference_buffer.getvalue()).decode('utf-8')
    boundaries_base64 = base64.b64encode(pil_boundaries_buffer.getvalue()).decode('utf-8')

    return JSONResponse(content={"Prediction": prediction, "Inference": inference_base64, "Boundaries": boundaries_base64})
