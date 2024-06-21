import mlflow.types
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow.types.schema as schema
from mlflow.types.schema import Schema, ColSpec
import os
import mlflow
import numpy as np
from scripts.preprocessing import preprocessing_for_training, preprocessing_for_testing

def build_model():#train_generator, validation_generator):

    # Ml Flow running on GCP Pre Check

    keyfile_path = '/mnt/airflow/keys/tensile-topic-424308-d9-7418db5a1c90.json'  # change as per your keyfile path

    # Checking if file exists
    if not os.path.exists(keyfile_path):
        raise FileNotFoundError(f"The file '{keyfile_path}' does not exist. Please check the path.")
    else:
        print(f'Path found : {keyfile_path}')

    # Set the environment variable to point to the service account key file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path
    os.environ['MLFLOW_GCS_BUCKET'] = 'ml-flow-remote-tracker-bucket'

    mlflow.set_tracking_uri("http://35.231.231.140:5000/")
    mlflow.set_experiment("Brain-Tumor-Classification")

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
        train_generator = preprocessing_for_training()
        validation_generator = preprocessing_for_testing(batchSize = 32)

        # Log the number of training and validation samples
        mlflow.log_param("num_train_samples", train_generator.samples)
        mlflow.log_param("num_val_samples", validation_generator.samples)

        # Train the model
        history = model.fit(train_generator, epochs=1, validation_data=validation_generator)

        #input_example = tf.random.uniform(shape=(1, *img_shape)).numpy()
        #output_example = model.predict(input_example) 
        #signature =  mlflow.models.signature.infer_signature(input_example, output_example)

        mlflow.keras.log_model(
            model,
            artifact_path="Brain_Tumor_Classification_Model",
            #input_example=input_example.numpy().tolist()
            #signature=signature
        )

        # Log the training metrics
        for epoch in range(1):
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("train_recall", history.history['recall'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_recall", history.history['val_recall'][epoch], step=epoch)


    return model


