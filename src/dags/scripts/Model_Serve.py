import mlflow
import os
import skimage.io
from mlflow import MlflowClient
from src.dags.scripts.preprocessing import load_and_preprocess_image
from src.dags.scripts.explainability import explain_inference

class Model_Server:

    def __init__(self, stage):
        self.stage = stage
        self._loadmodel()

    def _loadmodel(self):
        keyfile_path = '/home/p10/Documents/tensile-topic-424308-d9-17a256b9b21c.json'

        # Set the environment variable to point to the service account key file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path
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
        self.img_array = load_and_preprocess_image(img_path)
        self.preds = self.loaded_model.predict(self.img_array)
        #STORE IMAGE 
        return self.preds
    
    # Explain prediction made using LIME
    def explain_pred(self):
        return (explain_inference(self.img_array, self.loaded_model))
    






    



    
