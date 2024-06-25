import os
from airflow import DAG
from datetime import datetime
from airflow import configuration as conf
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.email_operator import EmailOperator
from scripts.retraining import check_feedback_size_flag, retrain_model
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------DAG------------------------------------------------------
conf.set('core','enable_xcom_pickling','True')

default_args = {
     "owner": "shaun",
     "retries" : 0,
     "start_date": datetime(2023, 12, 31)
     }

dag = DAG("Retraining_pipeline",
     schedule_interval = "@once", default_args = default_args)

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

# Define the email task
send_email = EmailOperator(
    task_id='send_email',
    to=os.getenv('EMAIL_TO'),    # Email address of the recipient
    subject='Notification from Airflow',
    html_content='<p>This is a notification email sent from Airflow indicating that the retraining dag was completd</p>',
    dag=dag,
    on_failure_callback=notify_failure,
    on_success_callback=notify_success
)


flag_true = EmailOperator(
    task_id = 'flag_true',
    to=os.getenv('EMAIL_TO'),    
    subject='Notification from Airflow for Retraining',
    html_content='<p>This is a notification email sent from Airflow indicating that the dag was triggered and retraining is in progress</p>',
    dag=dag
)

flag_false = EmailOperator(
    task_id = 'flag_false',
    to=os.getenv('EMAIL_TO'),   
    subject='Notification from Airflow for Retraining',
    html_content='<p>This is a notification email sent from Airflow indicating that the dag was triggered but stopped retraining stopped due to insuffucent files</p>',
    dag=dag
)

check_source_flag = BranchPythonOperator(
    task_id = 'check_source_flag',
    python_callable = check_feedback_size_flag,
    #op_args = [logger],
    dag = dag,
)

retrain_model = PythonOperator(
     task_id = 'retrain_model',
     python_callable = retrain_model,
     #op_args = [check_source_flag.output],
     dag = dag
)


check_source_flag >> flag_true >> retrain_model >> send_email
check_source_flag >> flag_false