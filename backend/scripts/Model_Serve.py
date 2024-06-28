import mlflow
import io
import skimage.io
# from mlflow import MlflowClient
from mlflow.tracking import MlflowClient
from scripts.preprocessing import load_and_preprocess_image
from scripts.explainability import explain_inference
from scripts.logger import setup_logging
import os
from PIL import Image
from google.cloud import storage
import uuid

class Model_Server:

    def __init__(self, stage):
        setup_logging("Object Created: Model_Server")
        # Set the environment variable to point to the service account key file
        #keyfile_path = "../backend/keys/tensile-topic-424308-d9-7418db5a1c90.json"
        keyfile_path = "../app/keys/tensile-topic-424308-d9-7418db5a1c90.json"
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path
        self.stage = stage
        self._loadmodel()

    def _loadmodel(self):
        
        model_name = 'Model_1_50_50'
        setup_logging("Method started: loadmodel")
        os.environ['MLFLOW_GCS_BUCKET'] = 'ml-flow-remote-tracker-bucket'

        mlflow.set_tracking_uri("http://35.231.231.140:5000/")
        mlflow.set_experiment("Brain-Tumor-Classification")

        client = MlflowClient()
        setup_logging(f"Loading model : {model_name}, Stage : {self.stage}")
        model_metadata = client.get_latest_versions(model_name, stages=[self.stage])

        if len(model_metadata)>1:
        
            # Access the metrics from the latest run
            run_id_latest = model_metadata[0].run_id   
            metrics_latest = client.get_run(run_id_latest).data.metrics

            # Access the metrics from the previous run
            run_id_prev = model_metadata[1].run_id   
            metrics_prev = client.get_run(run_id_prev).data.metrics

            if metrics_latest['val_recall'] > metrics_prev['val_recall']:
                logged_model = f'runs:/{run_id_latest}/model'
            else:
                logged_model = f'runs:/{run_id_prev}/model'

        else:
            logged_model = f'runs:/{model_metadata[0].run_id}/model'

        setup_logging(f"Loading model: {logged_model}")

        # Load model as a PyFuncModel.
        self.loaded_model = mlflow.pyfunc.load_model(logged_model)
        setup_logging("Method finished: loadmodel")
        

    def serve_model(self, img_path):
    
        # Load and make prediction
        self.img_array = load_and_preprocess_image(img_path)
        self.preds = self.loaded_model.predict(self.img_array)

        # Extract class info and create folder path to upload
        prediction_class = self._prediction(pred=self.preds)
        setup_logging(f"Serving model for image: {img_path} ; Prediction: {prediction_class}")

        folder_name = f'InferenceLogs/ImageLogs/{prediction_class}/'
        return prediction_class
    
    # Explain prediction made using LIME
    def explain_pred(self):
        setup_logging(f"Inference explained")
        return (explain_inference(self.img_array, self.loaded_model))
    
    
    def generate_unique_filename(self, existing_filenames, extension='jpg'):
        while True:
            random_name = f"{uuid.uuid4().hex}.{extension}"
            setup_logging(f"Random filename generated:{random_name}")
            if random_name not in existing_filenames:
                return random_name

    def get_existing_filenames(self, bucket_name="data-source-brain-tumor-classification"):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs()
        return [blob.name for blob in blobs]
    
    def uploadtobucket(self, file_path, file_name, folder_name, bucket_name = "data-source-brain-tumor-classification"):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob_name = os.path.join(folder_name, file_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        return True
        

    def move_file_in_bucket(self, file_name, source_folder, destination_folder, bucket_name="data-source-brain-tumor-classification"):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        source_blob_name = os.path.join(source_folder, file_name)
        destination_blob_name = os.path.join(destination_folder, file_name)
        
        source_blob = bucket.blob(source_blob_name)
        bucket.copy_blob(source_blob, bucket, destination_blob_name)
        source_blob.delete()

    def _prediction(self, pred):
        prediction = pred.argsort()[0, -5:][::-1][0]
        if prediction == 0: return "glioma"
        elif prediction == 1: return 'meningioma'
        elif prediction == 2: return 'notumor'
        elif prediction == 3: return 'pituitary'
