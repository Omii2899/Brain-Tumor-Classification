import os
from airflow import DAG
from datetime import datetime
from google.cloud import storage
from airflow import configuration as conf
from airflow.operators.python_operator import PythonOperator
from airflow.providers.google.cloud.hooks.gcs import gcs_object_is_directory
# from airflow.models import Connection
# from airflow import settings


# def create_connection():
#      new_conn = Connection(
#           conn_id="google_cloud_connection",
#           conn_type='google_cloud_platform',
#      )
#      scopes = [
#           "https://www.googleapis.com/auth/pubsub",
#           "https://www.googleapis.com/auth/datastore",
#           "https://www.googleapis.com/auth/bigquery",
#           "https://www.googleapis.com/auth/devstorage.read_write",
#           "https://www.googleapis.com/auth/logging.write",
#           "https://www.googleapis.com/auth/cloud-platform",
#      ]
#      conn_extra = {
#           "extra__google_cloud_platform__scope": ",".join(scopes),
#           "extra__google_cloud_platform__project": "tensile-topic-424308-d9",
#           "extra__google_cloud_platform__key_path": '/opt/airflow/tensile-topic-424308-d9-7418db5a1c90.json'
#      }
#      conn_extra_json = json.dumps(conn_extra)
#      new_conn.set_extra(conn_extra_json)

#      session = settings.Session()
#      if not (session.query(Connection).filter(Connection.conn_id == new_conn.conn_id).first()):
#           session.add(new_conn)
#           session.commit()
#      else:
#           msg = '\n\tA connection with `conn_id`={conn_id} already exists\n'
#           msg = msg.format(conn_id=new_conn.conn_id)
#           print(msg)

# def check_connection():
#     try:
#         credentials, project_id = google.auth.default()
#         if credentials is not None:
#             return {"credentials": credentials, "project_id": project_id}
#         else:
#             return False
#     except Exception as e:
#         return False

def check_source():
    return gcs_object_is_directory(bucket = "gs://data-source-brain-tumor-classification")


def download_blob(flag):
     if flag :
          bucket_name = "data-source-brain-tumor-classification"
          destination_folder = 'files'
          storage_client = storage.Client()

          bucket = storage_client.get_bucket(bucket_name)

          blobs = bucket.list_blobs()

          for blob in blobs:
               if blob.name.endswith('/'):
                    continue
               else:
                    destination_file_name = os.path.join(destination_folder, blob.name)

                    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
                    print(f"{blob.name} - {destination_file_name}")
                    blob.download_to_filename(destination_file_name)
     else:
         pass #Add Logging
# -------------------------------------DAG------------------------------------------------------
conf.set('core','enable_xcom_pickling','True')

default_args = {
     "owner": "aadarsh",
     "retries" : 5,
     "start_date": datetime(2023, 6, 6)
     }

dag = DAG("data_pipeline",
     schedule_interval = "@once", default_args = default_args,)

check_source = PythonOperator(
    task_id = "check_source",
    python_callable = check_source,
    dag = dag,
)

download_data = PythonOperator(
     task_id = 'download_data',
     python_callable = download_blob,
     op_args = [check_source.output],
     dag = dag,
)


check_source >> download_data


