import cv2
import os
import glob
import numpy as np
from scipy.stats import pearsonr
from google.cloud import storage
from scripts.logger import setup_logging
import pickle

BUCKET_NAME = "data-source-brain-tumor-classification"
HISTOGRAMS_FILE = 'validation/histograms.pkl'
#keyfile_path = 'keys/tensile-topic-424308-d9-7418db5a1c90.json' 
#keyfile_path = "../backend/keys/tensile-topic-424308-d9-7418db5a1c90.json"
#keyfile_path = "../app/keys/tensile-topic-424308-d9-7418db5a1c90.json"
#keyfile_path = './app/keys/tensile-topic-424308-d9-7418db5a1c90.json' 
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    setup_logging(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def validate_image(image):
    local_histogram_path = './histograms.pkl'

    try:
        download_from_gcs(BUCKET_NAME, HISTOGRAMS_FILE, local_histogram_path)
        with open(local_histogram_path, 'rb') as f:
            histograms = pickle.load(f)
        setup_logging("Loaded histograms from histograms.pkl")
    except Exception as e:
        setup_logging(f"Failed to load histograms: {e}", log_level='ERROR')
        return False
    setup_logging(f"Image: {image}")  
    load_image = cv2.imread(image)
    gray_image = cv2.cvtColor(load_image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    for hist in histograms:
        correlation = cv2.compareHist(histogram, hist, cv2.HISTCMP_CORREL)
        print(f"Image {image} has a reference histogram with correlation: {correlation:.4f}")
        if correlation > 0.7:
            return [True, correlation]
    setup_logging(f"Image validated: {correlation}")   
    setup_logging("Finished method: validate_image")   
    return [False, correlation]
