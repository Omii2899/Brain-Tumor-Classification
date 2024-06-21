import os
import mlflow
from google.cloud import storage
from Model_Serve import Model_Server
from preprocessing import download_files
from tensorflow.keras.optimizers import Adam
from preprocessing import preprocessing_for_testing, preprocessing_for_training

def check_feedback_size_flag(bucket_name = "data-source-brain-tumor-classification", size_flag = 50):
    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
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
        print()
        sum += folder_file_count[i]

    if sum < size_flag:
          return 'flag_false'
    else:
          return 'flag_true'


def retrain_model(flag):

    if flag==False:return False
    
    model = Model_Server().loaded_model

    download_files()
    
    #ML FLOW LOGGING

    # os.environ['MLFLOW_GCS_BUCKET'] = 'ml-flow-remote-tracker-bucket'

    # mlflow.set_tracking_uri("http://35.231.231.140:5000/")
    # mlflow.set_experiment("Brain-Tumor-Classification")

    optimizer = Adam(learning_rate=0.001, beta_1=0.85, beta_2=0.9925)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'recall'])

    model.fit(preprocessing_for_training(path = './InferenceLogs/ImageLogsWithFeedback/'), epochs=1, 
              validation_data=preprocessing_for_testing())


    


    



