import logging
import datetime
import os
from google.cloud import storage

def setup_logging(log_content, log_level='INFO'):
    keyfile_path = "/app/keys/tensile-topic-424308-d9-7418db5a1c90.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path
    object_name = 'logs/log_backend.txt'

    storage_client = storage.Client()
    bucket = storage_client.bucket('data-source-brain-tumor-classification')
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
