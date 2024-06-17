import tensorflow as tf
# from tensorflow.keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub 
import os
import mlflow.keras
import mlflow
from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import ImageDataGenerator 


# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './src/keys/tensile-topic-424308-d9-7418db5a1c90'

keyfile_path = './src/keys/akshita_keyfile.json'  #change as per your keyfile path
print(keyfile_path)

# Checking if file exists
if not os.path.exists(keyfile_path):
    raise FileNotFoundError(f"The file '{keyfile_path}' does not exist. Please check the path.")

# Set the environment variable to point to the service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path
os.environ['MLFLOW_GCS_BUCKET'] = 'ml-flow-remote-tracker-bucket'

mlflow.set_tracking_uri("http://35.231.231.140:5000/")
mlflow.set_experiment("Brain-Tumor-Classification")



# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     try:
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)
#         print("TensorFlow with Metal enabled with {} GPUs".format(len(physical_devices)))
#     except RuntimeError as e:
#         print(e)


def preprocessing_for_training():
    path = './data/Training/'
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    print("Training preprocessing done....")
    return train_generator

def preprocessing_for_testing_inference(batchSize):
    path = './data/Testing/'
    test_val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
    test_generator = test_val_datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=batchSize,
        class_mode='categorical',
        shuffle=False
    )
    print("Testing preprocessing done....")
    return test_generator


model = tf.keras.preprocessing.models.Sequential([
    tf.keras.preprocessing.layers .InputLayer(shape=(224, 224, 3)),
    tf.keras.preprocessing.layers .Conv2D(32, (3, 3), activation='relu'),
    tf.keras.preprocessing.layers .MaxPooling2D((2, 2)),
    tf.keras.preprocessing.layers .Conv2D(64, (3, 3), activation='relu'),
    tf.keras.preprocessing.layers .MaxPooling2D((2, 2)),
    tf.keras.preprocessing.layers .Conv2D(128, (3, 3), activation='relu'),
    tf.keras.preprocessing.layers .MaxPooling2D((2, 2)),
    tf.keras.preprocessing.layers .Flatten(),
    tf.keras.preprocessing.layers .Dense(128, activation='relu'),
    tf.keras.preprocessing.layers .Dense(4, activation='sigmoid')
])
model.summary()

train_dataset = preprocessing_for_training()
validation_dataset = preprocessing_for_testing_inference(batchSize=32)


with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss_function", "binary_crossentropy")
    mlflow.log_param("metrics", "accuracy")

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10
    )

    mlflow.keras.log_model(model, "model")

# Log metrics
for epoch in range(10):
    mlflow.log_metrics({
        "train_loss": history.history['loss'][epoch],
        "train_accuracy": history.history['accuracy'][epoch],
        "val_loss": history.history['val_loss'][epoch],
        "val_accuracy": history.history['val_accuracy'][epoch]
    }, step=epoch)


mlflow.log_artifacts('./data/Training/', artifact_path="training_data")
mlflow.log_artifacts('./data/Testing/', artifact_path="testing_data")