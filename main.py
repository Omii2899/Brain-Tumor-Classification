from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.pyfunc
import io
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Brain Tumor Classification API"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # preprocess
    image = image.resize(224,224)
    image_array = np.array(image)
    image_array = image_array * (1.0/255)

    # Getting the latest model from MLflow staging and loading it
    mlflow.set_tracking_uri("http://35.231.231.140:5000/")
    client = MlflowClient()
    model_metadata = client.get_latest_versions('Model_1_50_50', stages=["Staging"])
    model_version = model_metadata[0].version
    model_name = model_metadata[0].name
    model_uri = f"models:/{model_name}/{model_version}"
    print(model_uri)

    model = mlflow.pyfunc.load_model(model_uri)

    # Use the model to make a prediction
    prediction = model.predict(image_array)

    return JSONResponse(content={"prediction": prediction.tolist()})






    