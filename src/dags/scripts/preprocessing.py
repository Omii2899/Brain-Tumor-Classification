import os
import io
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scripts.logger import setup_logging 
from google.cloud import storage

def preprocessing_for_training(**kwargs):

    # Invoking the global logger method
    logger = setup_logging()
    logger.info("Started method: preprocessing_for_training")

    path = './data/Training/'
    logger.info(f"Image path: {path}")
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,           # Normalize pixel values to [0, 1]
        rotation_range=10,         # Rotate images up to 10 degrees
        width_shift_range=0.1,     # Shift images horizontally by up to 10% of width
        height_shift_range=0.1,    # Shift images vertically by up to 10% of height
        shear_range=0.1,           # Shear images by up to 10 degrees
        zoom_range=0.1,            # Zoom in or out by up to 10%
        horizontal_flip=True,      # Randomly flip images horizontally
        fill_mode='nearest'        # Use nearest pixel values to fill empty areas
    )

    train_generator = train_datagen.flow_from_directory(
    path,
    target_size=(224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True,
    seed = 42
    )

    logger.info("Finished method: preprocessing_for_training")
#     kwargs['ti'].xcom_push(key = 'train_generator', value = train_generator)
    return train_generator

def preprocessing_for_testing(batchSize, path= './data/Testing/',**kwargs):

     # Invoking the global logger method
     logger = setup_logging()
     logger.info("Started method: preprocessing_for_testing_inference")
     logger.info(f"Image path: {path}")
     logger.info(f'Batch size: {batchSize}')

     # Normalize pixel values to [0, 1]
     test_val_datagen = ImageDataGenerator(rescale=1.0/255)

     test_generator = test_val_datagen.flow_from_directory(
     path,
     target_size = (224, 224),
     batch_size = batchSize,
     class_mode = 'categorical',
     shuffle = False
     )

     logger.info("Finished method: preprocessing_for_testing_inference")
     return test_generator

def check_source():
     """
     Checks if the given GCS object (prefix) is a directory.
     
     :param bucket_name: Name of the GCS bucket.
     :param prefix: Prefix to check.
     :return: True if the prefix is a directory, False otherwise.
     """
     bucket_name = "data-source-brain-tumor-classification"
     logger = setup_logging()
     logger.info("Started Method: Check_Source")
     client = storage.Client()
     bucket = client.bucket(bucket_name)

     blobs = list(bucket.list_blobs())

     if len(blobs)>3:
          logger.info("Finished Method - Source Found")
          return True
     logger.warning("Finished Method - Source Not Found")
     return False


def download_files(flag):
     logger = setup_logging()
     logger.info("Method Started: Download_Files ")
     if flag :
          bucket_name = "data-source-brain-tumor-classification"
          destination_folder = ''
          storage_client = storage.Client()
          bucket = storage_client.get_bucket(bucket_name)
          blobs = bucket.list_blobs()
          for blob in blobs:
               if blob.name.endswith('/'):
                    continue
               else:
                    destination_file_name = os.path.join(destination_folder, blob.name)

                    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
                    # print(f"{blob.name} - {destination_file_name}")
                    blob.download_to_filename(destination_file_name)
          logger.info("Method Finished - Files Downloaded")


#Method to load and process image as an array
def load_and_preprocess_image(image_data, img_size=(224, 224, 3)):
#     #img = Image.open(io.BytesIO(image_data))
#     resized_img = image_data.resize(img_size)
#     #img = tf.keras.preprocessing.image.load_img(file_path, target_size=img_size)
#     img_array = tf.keras.preprocessing.image.img_to_array(resized_img)
#     img_array = np.expand_dims(img_array, 0)  # Create a batch
#     img_array /= 255.0  # Normalize to [0, 1]

    image = image_data.resize((224, 224))  # Resize the image
    image_array = np.array(image)
    image_array = image_array.astype('float32')  # Convert the image array to float32
    image_array = image_array * (1.0 / 255)  # Normalize the pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions to match model input shape

    return image_array

