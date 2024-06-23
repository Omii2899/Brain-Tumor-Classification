import os
import mlflow
from google.cloud import storage
from scripts.Model_Serve import Model_Server
from scripts.preprocessing import download_files
from tensorflow.keras.optimizers import Adam
from scripts.logger import setup_logging
from scripts.preprocessing import preprocessing_for_testing, preprocessing_for_training

def check_feedback_size_flag(bucket_name = "data-source-brain-tumor-classification", size_flag = 50):
    
    logger = setup_logging()
    logger.info("Started method: check_feedback_size_flag")
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

    logger.info(f"Number of files in feedback:{sum}")
    logger.info(f"Flag:{sum<size_flag}")
    logger.info("Method finished: check_feedback_size_flag")

    if sum < size_flag:
          return 'flag_false'
    else:
          return 'flag_true'


def retrain_model(flag):

    logger = setup_logging()
    logger.info("Method started: retrain_model")
    logger.info(f"Data Size Flag: {flag!=False}")

    if flag==False:return False
    
    model = Model_Server().loaded_model

    download_files()

    optimizer = Adam(learning_rate=0.001, beta_1=0.85, beta_2=0.9925)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'recall'])

    model.fit(preprocessing_for_training(path = './InferenceLogs/ImageLogsWithFeedback/'), epochs=1, 
              validation_data=preprocessing_for_testing())
    
    logger.info("Method finsihed: retrain_model")


    


    



