from datetime import datetime, timedelta  
from airflow import DAG  
from airflow.operators.dummy_operator import DummyOperator  
from airflow.operators.python_operator import PythonOperator

with DAG("forex_data_pipeline1", start_date = datetime(2024,6,1), 
     schedule_interval = "@daily", default_args = default_args, catchup = False) as dag:

    check_data_source = HttpSensor