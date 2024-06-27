from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.pyfunc
import os
import io
import base64
import time
import numpy as np
from scripts.Model_Serve import Model_Server
from scripts.statistics_histogram import validate_image
from scripts.logger import setup_logging
from scripts.postgres_connection import postgress_logger

app = FastAPI()
setup_logging("FastAPI application started")

ms = Model_Server(stage='Staging')
postgres_log = postgress_logger()



@app.get("/")
def read_root():
    setup_logging("Root endpoint accessed")
    return {"message": "Welcome to the Brain Tumor Classification API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    setup_logging("Predict endpoint accessed")
    contents = await file.read()
    start_time = time.time()
    image = Image.open(io.BytesIO(contents))
    existing_filenames = ms.get_existing_filenames()
    file_name = ms.generate_unique_filename(existing_filenames)        
    temp_file_path = f"/tmp/{file_name}"
    image.save(temp_file_path, format="JPEG")
    is_valid = validate_image(temp_file_path)
    postgres_log.image_path = file_name
    postgres_log.validation_flag = is_valid[0]
    postgres_log.correlation = is_valid[1]
    if not is_valid[0]:
        setup_logging(f"Invalid image uploaded: {file_name}")
        folder_name = 'InferenceLogs/ImageLogsForInvalidImages/'
        ms.uploadtobucket(temp_file_path, file_name, folder_name)
        postgres_log.flag = False
        postgres_log.push_to_postgres()
        return JSONResponse(content={"error": "Invalid Image. Please upload a correct brain MRI image."}, status_code=400)


    prediction = ms.serve_model(temp_file_path)
    pred_inf = ms.explain_pred()
    
    folder_name = f'InferenceLogs/ImageLogs/{prediction}/'
    ms.uploadtobucket(temp_file_path, file_name, folder_name)
    
    if postgres_log.flag == True:
        postgres_log.prediction = prediction
        postgres_log.glioma_probability = ms.preds[0][0]
        postgres_log.meningioma_probability = ms.preds[0][1]
        postgres_log.no_tumor_probability = ms.preds[0][2]
        postgres_log.pituitary_probability = ms.preds[0][3]
    
    pil_inference = Image.fromarray((pred_inf[0] * 255).astype(np.uint8))
    pil_boundaries = Image.fromarray((pred_inf[1] * 255).astype(np.uint8))

    pil_inference_buffer = io.BytesIO()
    pil_inference.save(pil_inference_buffer, format='JPEG')
    pil_boundaries_buffer = io.BytesIO()
    pil_boundaries.save(pil_boundaries_buffer, format='JPEG')

    inference_base64 = base64.b64encode(pil_inference_buffer.getvalue()).decode('utf-8')
    boundaries_base64 = base64.b64encode(pil_boundaries_buffer.getvalue()).decode('utf-8')
    setup_logging(f"Prediction made for file: {file_name}, prediction: {prediction}")
    postgres_log.flag = False
    postgres_log.time_taken = time.time() - start_time
    postgres_log.push_to_postgres()
    return JSONResponse(content={"Prediction": prediction, "Inference": inference_base64, "Boundaries": boundaries_base64, "FileName": file_name})

@app.post("/feedback/")
async def feedback(file_name: str = Form(...), corrected_label: str = Form(...), prediction: str = Form(...)):
    original_folder = f'InferenceLogs/ImageLogs/{prediction}/'
    feedback_folder = f'InferenceLogs/ImageLogsWithFeedback/{corrected_label}/'
    postgres_log.image_path = file_name
    postgres_log.feedback_flag = True
    postgres_log.feedback_class = corrected_label
    postgres_log.push_feedback_postgres()
    ms.move_file_in_bucket(file_name, original_folder, feedback_folder)
    setup_logging(f"Feedback recorded for file: {file_name}, corrected label: {corrected_label}")
    return JSONResponse(content={})