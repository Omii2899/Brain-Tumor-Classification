import logging
import datetime
import os
from dotenv import load_dotenv
from google.cloud import storage

def setup_logging(log_content, log_level='INFO'):
    load_dotenv()
    object_name = 'logs/log_dags.txt'

    storage_client = storage.Client()
    bucket = storage_client.bucket(os.getenv('BUCKET_NAME'))
    blob = bucket.get_blob(object_name)

    if not blob:
    # Create the file if it doesn't exist
        blob = bucket.blob(object_name)

    dataT=""
    with bucket.blob(object_name).open('r') as f:
        dataT=f.read()

    # Generate current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"\n[{timestamp}]: {log_level} - {log_content}\n"
    
    #append in that str
    dataT = dataT + log_entry
    
    #write that str back
    with bucket.blob(object_name).open('w') as f:
        f.write(dataT)
