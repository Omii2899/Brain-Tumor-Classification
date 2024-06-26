import mlflow.types
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow.types.schema as schema
from mlflow.types.schema import Schema, ColSpec
from scripts.logger import setup_logging
from scripts.preprocessing import preprocessing_for_testing, preprocessing_for_training
import os
import mlflow
import numpy as np
from dotenv import load_dotenv


def build_model():#train_generator, validation_generator):

    load_dotenv()
    
    #logger = setup_logging()
    setup_logging("Started method: Building Model")
    #keyfile_path = 'keys/tensile-topic-424308-d9-7418db5a1c90.json'  # change as per your keyfile path
    keyfile_path = os.getenv('KEYFILE_PATH')

    # Checking if file exists
    if not os.path.exists(keyfile_path):
        raise FileNotFoundError(f"The file '{keyfile_path}' does not exist. Please check the path.")
    else:
        print(f'Path found : {keyfile_path}')

    # Set the environment variable to point to the service account key file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path
    os.environ['MLFLOW_GCS_BUCKET'] = os.getenv('MLFLOW_BUCKET')

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URL'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT'))

    num_classes = 4
    img_shape = (224, 224, 3)

    with mlflow.start_run():
        
        # Create the model
        model = Sequential([
            Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=img_shape),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),

            Dense(512, activation="relu"),
            Dense(num_classes, activation="softmax")
        ])

        # Log the model summary
        model.summary(print_fn=lambda x: mlflow.log_text(x, "model_summary.txt"))

        # Log the model configuration
        mlflow.log_param("model_type", "Sequential")
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("img_shape", img_shape)

        optimizer = Adam(learning_rate=0.001, beta_1=0.85, beta_2=0.9925)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'recall'])

        # Log the optimizer configuration
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 0.001)\
        
        mlflow.log_param("beta_1", 0.85)
        mlflow.log_param("beta_2", 0.9925)

        # Prepare data generators
        
        path = './data/Training/'

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
    
        test_val_datagen = ImageDataGenerator(rescale=1.0/255)

        test_generator = test_val_datagen.flow_from_directory(
        path,
        target_size = (224, 224),
        batch_size = 32,
        class_mode = 'categorical',
        shuffle = False
        )

        # Log the number of training and validation samples
        mlflow.log_param("num_train_samples", train_generator.samples)
        mlflow.log_param("num_val_samples", test_generator.samples)

        # Train the model
        history = model.fit(train_generator, epochs=1, validation_data=test_generator)
        setup_logging("Finished training model")

        # Log the training metrics
        for epoch in range(1):
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("train_recall", history.history['recall'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_recall", history.history['val_recall'][epoch], step=epoch)
        
    
    setup_logging("Finished method: Build Model")
    return model

def build_hp_model(hp):

    load_dotenv()
    keyfile_path = os.getenv('KEYFILE_PATH')

    # Set the environment variable to point to the service account key file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path
    os.environ['MLFLOW_GCS_BUCKET'] = os.getenv('MLFLOW_BUCKET')

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URL'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT'))

    with mlflow.start_run(nested=True): 
        model = Sequential()
        model.add(tf.keras.layers.Input(shape=(224, 224, 3)))  # Correctly initialize the input shape

        for i in range(1, 3):
            filters = hp.Int(f'conv_{i}_filters', min_value=32, max_value=256, step=32)
            model.add(Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                activation='relu',
                padding='same'
            ))
            mlflow.log_param(f'conv_{i}_filters', filters)
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        num_dense_layers = hp.Int('num_dense_layers', 1, 3)
        mlflow.log_param('num_dense_layers', num_dense_layers)

        for i in range(num_dense_layers):
            units = hp.Int(f'dense_{i}_units', min_value=128, max_value=512, step=128)
            model.add(Dense(units=units, activation='relu'))
            mlflow.log_param(f'dense_{i}_units', units)
            if hp.Boolean(f'dropout_{i}'):
                dropout_rate = hp.Float(f'dropout_{i}_rate', min_value=0.2, max_value=0.5, step=0.1)
                model.add(Dropout(rate=dropout_rate))
                mlflow.log_param(f'dropout_{i}_rate', dropout_rate)

        model.add(Dense(4, activation='softmax'))

        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        mlflow.log_param('learning_rate', learning_rate)

        tuner = kt.Hyperband(
            build_model,
            objective='val_accuracy',
            max_epochs=10,
            factor=3
            # directory='./model_runs/',
            # project_name='brain_tumor_classification'
        )

        with mlflow.start_run():
            mlflow.keras.autolog()
            tuner.search(preprocessing_for_training(), epochs=50, validation_data=preprocessing_for_testing(32))

     

        



