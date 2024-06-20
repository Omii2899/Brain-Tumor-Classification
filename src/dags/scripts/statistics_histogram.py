import cv2
import pickle
import os
import glob
from logger import setup_logging

# Method to capture statistics of each image in each class
def capture_histograms():

    # Invoking the global logger method
    logger = setup_logging()
    logger.info("Started method: capture_histograms")

    # Info about base directory to access images and there respective classes
    base_dir = './data/Training/'
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    histograms = []

    # Picking random images from each class and capturing there histograms
    for cls in classes:
        folder_path = os.path.join(base_dir, cls)
        image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        
        if image_paths:
            first_image_path = image_paths[0]
            logger.info(f"Processing image: {first_image_path}")
            try:
                load_image = cv2.imread(first_image_path)
                gray_image = cv2.cvtColor(load_image, cv2.COLOR_BGR2GRAY)
                histograms.append(cv2.calcHist([gray_image], [0], None, [256], [0, 256]))
                logger.info("Captured histogram successfully")
            except Exception as e:
                logger.exception(f"Error processing image {first_image_path}: {e}")
        else:
            logger.warning(f"No images found in directory: {folder_path}")
            return False
    
    logger.info("Finished capturing statistics")

    # Saving the histogram info for inference 
    try:
        with open('./src/histograms.pkl', 'wb') as f:
            pickle.dump(histograms, f)
        logger.info("Histogram data saved successfully to histograms.pkl")
    except Exception as e:
        logger.exception(f"Error saving histogram data: {e}")
        return False
    
    logger.info("Finished method: capture_histograms")
    return True


def validate_image(image):

    # Invoking the global logger method
    logger = setup_logging()
    logger.info("Started method: validate_image")

    # Path to saved histograms
    histograms_path = 'histograms.pkl'

    # Loading histograms saved by capture_histograms
    try:
        with open(histograms_path, 'rb') as f:
            histograms = pickle.load(f)
        logger.info("Loaded histograms from histograms.pkl")
    except Exception as e:
        logger.exception(f"Failed to load histograms: {e}")
        return False
    
    # Load the image passed and compare it with the histograms of different classes
    load_image = cv2.imread(image)
    gray_image = cv2.cvtColor(load_image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    for hist in histograms:
        correlation = cv2.compareHist(histogram, hist, cv2.HISTCMP_CORREL)
        logger.info(f"Image {image} has a reference histogram with correlation: {correlation:.4f}")
        if correlation > 0.8:
          return True

    logger.info("Finished method: validate_image")   
    return False

capture_histograms()



