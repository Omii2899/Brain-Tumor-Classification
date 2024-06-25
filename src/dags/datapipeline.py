import os
from airflow import DAG
from dotenv import load_dotenv, dotenv_values
from datetime import datetime
from google.cloud import storage
from airflow import configuration as conf
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from scripts.logger import setup_logging
from scripts.preprocessing import preprocessing_for_testing, preprocessing_for_training, check_source, download_files
from scripts.statistics_histogram import capture_histograms
from scripts.model_trainer import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from airflow.models import XCom


load_dotenv()

# Define function to notify failure or sucess via an email
def notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to=os.getenv('EMAIL_TO'),
        subject='Success Notification from Airflow',
        html_content='<p>The dag tasks succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def notify_failure(context):
    failure_email = EmailOperator(
        task_id='failure_email',
        to=os.getenv('EMAIL_TO'),
        subject='Failure Notification from Airflow',
        html_content='<p>The dag tasks failed.</p>',
        dag=context['dag']
    )
    failure_email.execute(context=context)
# -------------------------------------DAG------------------------------------------------------
conf.set('core','enable_xcom_pickling','True')

default_args = {
     "owner": "aadarsh",
     "retries" : 0,
     "start_date": datetime(2023, 12, 31)
     }

dag = DAG("data_pipeline",
     schedule_interval = "@once", default_args = default_args)

# Define the email task
send_email = EmailOperator(
    task_id='send_email',
    to=os.getenv('EMAIL_TO'),    # Email address of the recipient
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
     python_callable = preprocessing_for_testing,
     op_args = [32], # path,batch size
     dag = dag
)

train_model = PythonOperator(
     task_id = 'builiding_model',
     python_callable = build_model,
     #op_args = [augment_transform_training_data.output, transform_testing_data.output],
     # op_args = ["{{ti.xcom_pull(key='train_generator', task_ids='augment_transform_training_data')}}",
     #            "{{ti.xcom_pull(key='test_generator', task_ids=''transform_testing_data')}}"], 
     dag = dag
)

#check_source >> download_data >> capture_statistics >> augment_transform_training_data >> transform_testing_data

check_source >> download_data >> capture_statistics >> [augment_transform_training_data, transform_testing_data] >> train_model >> send_email
