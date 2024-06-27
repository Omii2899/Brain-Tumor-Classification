import cv2
import pickle
import os
import glob
import numpy as np
from scipy.stats import pearsonr
from google.cloud import storage
from scripts.logger import setup_logging
from dotenv import load_dotenv, dotenv_values

load_dotenv()
BUCKET_NAME = os.getenv('BUCKET_NAME')
HISTOGRAMS_FILE = os.getenv('HISTOGRAMS_FILE_PATH_GCP')
keyfile_path = os.getenv('KEYFILE_PATH') 
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path

def upload_to_gcs(bucket_name, destination_blob_name, source_file_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    setup_logging(f"File {source_file_name} uploaded to {destination_blob_name}.")

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    setup_logging(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def capture_histograms():
    base_dir = './data/Training/'
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    #logger = setup_logging()
    
    histograms = []

    for cls in classes:
        folder_path = os.path.join(base_dir, cls)
        image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        
        if image_paths:
            first_image_path = image_paths[0]
            try:
                first_image = cv2.imread(first_image_path)
                gray_first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
                hist_first_image = cv2.calcHist([gray_first_image], [0], None, [256], [0, 256])

                valid_histograms = [hist_first_image]

                for image_path in image_paths[1:]:
                    try:
                        load_image = cv2.imread(image_path)
                        gray_image = cv2.cvtColor(load_image, cv2.COLOR_BGR2GRAY)
                        hist_image = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

                        correlation = pearsonr(hist_first_image.flatten(), hist_image.flatten())[0]

                        if correlation > 0.7:
                            valid_histograms.append(hist_image)

                    except Exception as e:
                        setup_logging(f"Error processing image {image_path}: {e}", log_level='ERROR')

                histograms.extend(valid_histograms)
                setup_logging(f"Captured {len(valid_histograms)} histograms for class {cls}")

            except Exception as e:
                setup_logging(f"Error processing images in folder {folder_path}: {e}", log_level="EXCEPTION")
        else:
            setup_logging(f"No images found in directory: {folder_path}")
            return False
    
    setup_logging("Finished capturing histograms")

    try:
        local_histogram_path = './histograms.pkl'
        with open(local_histogram_path, 'wb') as f:
            pickle.dump(histograms, f)
        print("Histogram data saved locally to histograms.pkl")

        upload_to_gcs(BUCKET_NAME, HISTOGRAMS_FILE, local_histogram_path)
        setup_logging("Histogram data uploaded to GCS")

    except Exception as e:
        setup_logging(f"Error saving histogram data: {e}", log_level='ERROR')
        return False
    
    setup_logging("Finished method: capture_histograms")
    return True

def validate_image(image):
    local_histogram_path = './histograms.pkl'

    try:
        download_from_gcs(BUCKET_NAME, HISTOGRAMS_FILE, local_histogram_path)
        with open(local_histogram_path, 'rb') as f:
            histograms = pickle.load(f)
        setup_logging("Loaded histograms from histograms.pkl")
    except Exception as e:
        setup_logging(f"Failed to load histograms: {e}")
        return False
    
    load_image = cv2.imread(image)
    gray_image = cv2.cvtColor(load_image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    for hist in histograms:
        correlation = cv2.compareHist(histogram, hist, cv2.HISTCMP_CORREL)
        setup_logging(f"Image {image} has a reference histogram with correlation: {correlation:.4f}")
        if correlation > 0.7:
            return True

    setup_logging("Finished method: validate_image")   
    return False

