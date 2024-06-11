import os
from airflow import DAG
from datetime import datetime
from google.cloud import storage
from airflow import configuration as conf
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
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



default_args = {
     "owner": "aadarsh",
     "retries" : 0,
     "start_date": datetime(2023, 12, 31)
     }

dag = DAG("data_pipeline",
     schedule_interval = "@once", default_args = default_args)


# Define function to notify failure or sucess via an email
def notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to='sharma.yasha@northeastern.edu',
        subject='Success Notification from Airflow',
        html_content='<p>The dag tasks succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def notify_failure(context):
    failure_email = EmailOperator(
        task_id='failure_email',
        to='sharma.yasha@northeastern.edu',
        subject='Failure Notification from Airflow',
        html_content='<p>The dag tasks failed.</p>',
        dag=context['dag']
    )
    failure_email.execute(context=context)

# Define the email task
send_email = EmailOperator(
    task_id='send_email',
    to='sharma.yasha@northeastern.edu',    # Email address of the recipient
    subject='Notification from Airflow',
    html_content='<p>This is a notification email sent from Airflow indicating that the dag was triggered</p>',
    dag=dag,
    on_failure_callback=notify_failure,
    on_success_callback=notify_success
)


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


#check_source >> download_data >> capture_statistics >> augment_transform_training_data >> transform_testing_data

check_source >> download_data >> capture_statistics
capture_statistics >> [augment_transform_training_data, transform_testing_data] >> send_email





