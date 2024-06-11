import os
from airflow import DAG
from datetime import datetime
from google.cloud import storage
from airflow import configuration as conf
from airflow.operators.python_operator import PythonOperator
from scripts.logger import setup_logging
from scripts.preprocessing import preprocessing_for_testing_inference, preprocessing_for_training
from scripts.statistics import capture_histograms


def check_source():
     """
     Checks if the given GCS object (prefix) is a directory.
     
     :param bucket_name: Name of the GCS bucket.
     :param prefix: Prefix to check.
     :return: True if the prefix is a directory, False otherwise.
     """
     bucket_name = "data-source-brain-tumor-classification"
     logger = setup_logging()
     logger.info("Started Method: Check_Source")
     client = storage.Client()
     bucket = client.bucket(bucket_name)

     blobs = list(bucket.list_blobs())

     if len(blobs)>3:
          logger.info("Finished Method - Source Found")
          return True
     logger.warning("Finished Method - Source Not Found")
     return False

def download_files(flag):
     logger = setup_logging()
     logger.info("Method Started: Download_Files ")
     if flag :
          bucket_name = "data-source-brain-tumor-classification"
          destination_folder = ''
          storage_client = storage.Client()
          bucket = storage_client.get_bucket(bucket_name)
          blobs = bucket.list_blobs()
          for blob in blobs:
               if blob.name.endswith('/'):
                    continue
               else:
                    destination_file_name = os.path.join(destination_folder, blob.name)

                    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
                    # print(f"{blob.name} - {destination_file_name}")
                    blob.download_to_filename(destination_file_name)
          logger.info("Method Finished - Files Downloaded")


# -------------------------------------DAG------------------------------------------------------
conf.set('core','enable_xcom_pickling','True')

# Invoking the global logger method
logger = setup_logging()
logger.info("Started DAG pipeline: datapipeline")

default_args = {
     "owner": "aadarsh",
     "retries" : 1,
     "start_date": datetime(2023, 6, 6)
     }

dag = DAG("data_pipeline",
     schedule_interval = "@once", default_args = default_args)


check_source = PythonOperator(
    task_id = "check_source",
    python_callable = check_source,
    #op_args = [logger],
    dag = dag,
)

download_data = PythonOperator(
     task_id = 'download_data',
     python_callable = download_files,
     op_args = [check_source.output],
     dag = dag,
)


capture_statistics = PythonOperator(
     task_id = 'capture_statistics',
     python_callable = capture_histograms,
     #op_args = [logger],
     dag = dag
)

augment_transform_training_data = PythonOperator(
     task_id = 'augment_input_data',
     python_callable = preprocessing_for_training,
     #op_args = [logger],
     dag = dag
)

transform_testing_data = PythonOperator(
     task_id = 'transform_testing_data',
     python_callable = preprocessing_for_testing_inference,
     op_args = ['./data/Testing', 32], # path,batch size
     dag = dag
     
)


check_source >> download_data >> capture_statistics >> augment_transform_training_data >> transform_testing_data


