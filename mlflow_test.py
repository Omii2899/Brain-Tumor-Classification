from mlflow.tracking import MlflowClient
import mlflow
import mlflow.pyfunc
import io
import numpy as np
from PIL import Image
import os


keyfile_path = 'src/keys/tensile-topic-424308-d9-7418db5a1c90.json'  #change as per your keyfile path

print(keyfile_path)

# Checking if file exists
if not os.path.exists(keyfile_path):
    raise FileNotFoundError(f"The file '{keyfile_path}' does not exist. Please check the path.")

# Set the environment variable to point to the service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path
# Add this line to read an image from a folder for testing purposes
test_image_path = "Te-no_0037.jpg"  # Replace with your actual image path

# Use this line to read the image from the folder
image = Image.open(test_image_path)

# Preprocess the image
image = image.resize((224, 224))  # Corrected resize tuple
image_array = np.array(image)
image_array = image_array.astype('float32')
image_array = image_array * (1.0/255)
image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions to match model input shape

# Getting the latest model from MLflow staging and loading it
mlflow.set_tracking_uri("http://35.231.231.140:5000/")
client = MlflowClient()
model_metadata = client.get_latest_versions('Main', stages=["Staging"])
model_version = model_metadata[0].version
model_name = model_metadata[0].name
model_uri = f"models:/{model_name}/{model_version}"

model = mlflow.pyfunc.load_model(model_uri)

# Use the model to make a prediction
prediction = model.predict(image_array)

print(prediction)
