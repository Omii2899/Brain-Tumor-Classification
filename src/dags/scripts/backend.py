from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.pyfunc
import os
import io
import numpy as np
from Model_Serve import Model_Server


# # Set the path to your service account key file
# keyfile_path = '../../keys/tensile-topic-424308-d9-7418db5a1c90.json'  # change as per your keyfile path

# # Checking if the file exists
# if not os.path.exists(keyfile_path):
#     raise FileNotFoundError(f"The file '{keyfile_path}' does not exist. Please check the path.")

# # Set the environment variable to point to the service account key file
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path

app = FastAPI()

ms = Model_Server(stage='Staging')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Brain Tumor Classification API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()
    file_name = file.filename
    temp_file_path = f"/tmp/{file_name}"  # Temporary file path
    image = Image.open(io.BytesIO(contents))
    image.save(temp_file_path, format="JPEG")
    #Make prediction
    prediction = ms.serve_model(image)
    folder_name = f'InferenceLogs/ImageLogs/{prediction}/'
    ms.uploadtobucket(temp_file_path, file_name, folder_name)
    
    # # Preprocess the image
    # image = image.resize((224, 224))  # Resize the image
    # image_array = np.array(image)
    # image_array = image_array.astype('float32')  # Convert the image array to float32
    # image_array = image_array * (1.0 / 255)  # Normalize the pixel values
    # image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions to match model input shape

    # # Getting the latest model from MLflow staging and loading it
    # mlflow.set_tracking_uri("http://35.231.231.140:5000/")
    # client = MlflowClient()
    # model_metadata = client.get_latest_versions('Main', stages=["Staging"])
    # model_version = model_metadata[0].version
    # model_name = model_metadata[0].name
    # model_uri = f"models:/{model_name}/{model_version}"

    # model = mlflow.pyfunc.load_model(model_uri)

    # Use the model to make a prediction
    # prediction = model.predict(image_array)

    # return JSONResponse(content={"prediction": prediction.tolist()})
    return JSONResponse(content={"prediction": prediction})
