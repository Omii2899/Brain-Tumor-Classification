import airflow


with DAG("forex_data_pipeline1", start_date = datetime(2024,6,1), 
     schedule_interval = "@daily", default_args = default_args, catchup = False) as dag:

    check_data_source = HttpSensor