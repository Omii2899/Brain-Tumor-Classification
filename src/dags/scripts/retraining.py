import os
import mlflow
import mlflow.pyfunc
import tensorflow as tf
from mlflow.tracking import MlflowClient
from google.cloud import storage
from scripts.Model_Serve import Model_Server
from scripts.preprocessing import download_files
from tensorflow.keras.optimizers import Adam
from scripts.logger import setup_logging
from scripts.preprocessing import preprocessing_for_testing, preprocessing_for_training
from dotenv import load_dotenv



def check_feedback_size_flag(bucket_name =os.getenv('BUCKET_NAME'), size_flag = 5):
    load_dotenv()
    keyfile_path = os.getenv('KEYFILE_PATH')
    # Checking if file exists
    if not os.path.exists(keyfile_path):
        setup_logging(f"The file '{keyfile_path}' does not exist. Please check the path.", log_level='ERROR')

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path
    #logger = setup_logging()
    setup_logging("Started method: check_feedback_size_flag")
    storage_client = storage.Client()
    
    bucket = storage_client.get_bucket(os.getenv('BUCKET_NAME'))
    blobs = bucket.list_blobs(prefix='InferenceLogs/ImageLogsWithFeedback/')    
    
    # Dictionary to store folder and file count and var to stor numbe rof files
    folder_file_count = {}
    sum=-0

    for blob in blobs:
        # Get the folder path
        if blob.name.endswith('/'):
                        continue
        folder_path = '/'.join(blob.name.split('/')[:-1])
        if folder_path not in folder_file_count:
            folder_file_count[folder_path] = 0
        folder_file_count[folder_path] += 1
    
    for i in folder_file_count:
        sum += folder_file_count[i]

    setup_logging(f"Number of files in feedback:{sum}")
    # setup_logging(f"Count of files:{sum}")
    setup_logging("Method finished: check_feedback_size_flag")

    if sum < size_flag:
          return 'flag_false'
    else:
          return 'flag_true'



# def retrain_model():

#     #logger = setup_logging()
#     setup_logging("Method started: retrain_model")
#     setup_logging(f"Data Size Flag: {flag!=False}")

#     if flag==False:return False
    
#     #model = Model_Server().loaded_model
#     logged_model = 'runs:/8d92ccb5d21e42798bcced8b7ea384eb/Brain_Tumor_Classification_Model'

#     # Load model as a PyFuncModel.
#     model = mlflow.pyfunc.load_model(logged_model)

#     download_files()

#     optimizer = Adam(learning_rate=0.001, beta_1=0.85, beta_2=0.9925)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'recall'])

#     model.fit(preprocessing_for_training(path = './InferenceLogs/ImageLogsWithFeedback/'), epochs=1, 
#               validation_data=preprocessing_for_testing())
    
#     setup_logging("Method finsihed: retrain_model")

def retrain_model():
    setup_logging("Method started: retrain_model")
    #setup_logging(f"Data Size Flag: {flag != False}")
    os.environ['MLFLOW_GCS_BUCKET'] = os.getenv('MLFLOW_BUCKET')

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URL'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT'))

    #if flag == False:return False
    
    # Log parameters using MLflow
    with mlflow.start_run() as run:
        #mlflow.log_param("flag", flag)

        logged_model = 'runs:/8d92ccb5d21e42798bcced8b7ea384eb/Brain_Tumor_Classification_Model'
        try:
            model = mlflow.pyfunc.load_model(logged_model)
            setup_logging(f"Model loaded:{logged_model}")
        except AttributeError as e:
            setup_logging(f"AttributeError encountered: {e}", log_level='ERROR')
            setup_logging("Failed to load model or execute training.", log_level='ERROR')
            return False

        # Download necessary files
        storage_client = storage.Client()
        bucket = storage_client.bucket(os.getenv('BUCKET_NAME'))
        blob = bucket.blob('InferenceLogs/ImageLogsWithFeedback/')
        blob.download_to_filename('./')

        optimizer = Adam(learning_rate=0.001, beta_1=0.85, beta_2=0.9925)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'recall'])

        train_data = preprocessing_for_training(path='./InferenceLogs/ImageLogsWithFeedback/')
        validation_data = preprocessing_for_testing()
        history = model.fit(train_data, epochs=1, validation_data=validation_data)

        mlflow.log_metrics({
            'train_loss': history.history['loss'][0],
            'train_accuracy': history.history['accuracy'][0],
            'val_loss': history.history['val_loss'][0],
            'val_accuracy': history.history['val_accuracy'][0],
            'val_recall': history.history['val_recall'][0]
        })

        setup_logging("Method finished: retrain_model")

    run_id = run.info.run_id
    setup_logging(f"MLflow run ID: {run_id}")

    return True


    





    


    



