import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scripts.logger import setup_logging 

def preprocessing_for_training():

    # Invoking the global logger method
    logger = setup_logging()
    logger.info("Started method: preprocessing_for_training")

    path = './data/Training/'
    logger.info(f"Image path: {path}")
    
    train_datagen = ImageDataGenerator(
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
    #return train_generator

def preprocessing_for_testing_inference(path, batchSize):

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
    #return test_generator

