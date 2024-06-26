from fastapi import FastAPI, File, UploadFile, Form
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
from scripts.logger import setup_logging

app = FastAPI()
setup_logging("FastAPI application started")

ms = Model_Server(stage='Staging')

@app.get("/")
def read_root():
    setup_logging("Root endpoint accessed")
    return {"message": "Welcome to the Brain Tumor Classification API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    setup_logging("Predict endpoint accessed")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    existing_filenames = ms.get_existing_filenames()
    file_name = ms.generate_unique_filename(existing_filenames)
    temp_file_path = f"/tmp/{file_name}"
    image.save(temp_file_path, format="JPEG")
    is_valid = validate_image(temp_file_path)

    if not is_valid:
        setup_logging(f"Invalid image uploaded: {file_name}", log_level='ERROR')
        folder_name = 'InferenceLogs/ImageLogsForInvalidImages/'
        ms.uploadtobucket(temp_file_path, file_name, folder_name)
        return JSONResponse(content={"error": "Invalid Image. Please upload a correct brain MRI image."}, status_code=400)

    prediction = ms.serve_model(temp_file_path)
    pred_inf = ms.explain_pred()

    folder_name = f'InferenceLogs/ImageLogs/{prediction}/'
    ms.uploadtobucket(temp_file_path, file_name, folder_name)
    
    pil_inference = Image.fromarray((pred_inf[0] * 255).astype(np.uint8))
    pil_boundaries = Image.fromarray((pred_inf[1] * 255).astype(np.uint8))

    pil_inference_buffer = io.BytesIO()
    pil_inference.save(pil_inference_buffer, format='JPEG')
    pil_boundaries_buffer = io.BytesIO()
    pil_boundaries.save(pil_boundaries_buffer, format='JPEG')

    inference_base64 = base64.b64encode(pil_inference_buffer.getvalue()).decode('utf-8')
    boundaries_base64 = base64.b64encode(pil_boundaries_buffer.getvalue()).decode('utf-8')
    setup_logging(f"Prediction made for file: {file_name}, prediction: {prediction}")
    return JSONResponse(content={"Prediction": prediction, "Inference": inference_base64, "Boundaries": boundaries_base64, "FileName": file_name})

@app.post("/feedback/")
async def feedback(file_name: str = Form(...), corrected_label: str = Form(...), prediction: str = Form(...)):
    original_folder = f'InferenceLogs/ImageLogs/{prediction}/'
    feedback_folder = f'InferenceLogs/ImageLogsWithFeedback/{corrected_label}/'

    ms.move_file_in_bucket(file_name, original_folder, feedback_folder)
    setup_logging(f"Feedback recorded for file: {file_name}, corrected label: {corrected_label}")
    #return JSONResponse(content={"message": "Feedback recorded and image moved successfully."})
    return JSONResponse(content={})