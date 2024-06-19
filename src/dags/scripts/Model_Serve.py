import mlflow
import os
import skimage.io
from mlflow import MlflowClient
from src.dags.scripts.preprocessing import load_and_preprocess_image
from src.dags.scripts.explainability import explain_inference
import os
from google.cloud import storage

class Model_Server:

     
    

    def __init__(self, stage):
        # Set the environment variable to point to the service account key file
        keyfile_path = '../../keys/tensile-topic-424308-d9-7418db5a1c90.json' 
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path
        self.stage = stage
        self._loadmodel()

    def _loadmodel(self):

        os.environ['MLFLOW_GCS_BUCKET'] = 'ml-flow-remote-tracker-bucket'

        mlflow.set_tracking_uri("http://35.231.231.140:5000/")
        mlflow.set_experiment("Brain-Tumor-Classification")

        client = MlflowClient()
        model_metadata = client.get_latest_versions('Model_1_50_50', stages=[self.stage])

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
        #logged_model = 'runs:/33047b43cbe9447e94972cd460d768da/Brain_Tumor_Classification_Model'
        #logged_model = 'runs:/c5c9b41a3163479e9e1fd4a6a8384e31/model'

        # Load model as a PyFuncModel.
        self.loaded_model = mlflow.pyfunc.load_model(logged_model)
        

    def serve_model(self, img_path):

        # Load and make prediction
        self.img_array = load_and_preprocess_image(img_path)
        preds = self.loaded_model.predict(self.img_array)

        # Extract class info and create folder path to upload
        prediction_class = self._prediction(pred=preds)
        folder_name = f'InferenceLogs/ImageLogs/{prediction_class}/'

        #Upload inference image to logs
        self.uploadtobucket(img_path, folder_name)
    
        return prediction_class
    
    # Explain prediction made using LIME
    def explain_pred(self):
        return (explain_inference(self.img_array, self.loaded_model))
    

    def uploadtobucket(self, img_path, folder_name, bucket_name = "data-source-brain-tumor-classification"):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        # Use the file name with folder path as the blob name
        blob_name = os.path.join(folder_name, os.path.basename(img_path))
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(img_path)
        return True
    
    def _prediction(self, pred):
        prediction = pred.argsort()[0, -5:][::-1][0]
        if prediction == 0: return "giloma"
        elif prediction == 1: return 'meningioma'
        elif prediction == 2: return 'notumor'
        elif prediction == 3: return 'pituitary'
        







    



    
