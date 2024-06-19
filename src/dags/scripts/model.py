<<<<<<< HEAD
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub 
import os
import mlflow.keras
import mlflow


# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './src/keys/tensile-topic-424308-d9-7418db5a1c90'

#keyfile_path = '../../keys/akshita_keyfile.json'/home/yashasvi/Desktop/MLOPS/Project/Brain-Tumor-Classification/src/keys/tensile-topic-424308-d9-7418db5a1c90.json  #change as per your keyfile path
keyfile_path = '../../keys/tensile-topic-424308-d9-7418db5a1c90.json'

# Checking if file exists
=======
# import tensorflow as tf
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from google.cloud import storage
from google.oauth2 import service_account
import os

keyfile_path = '../../keys/akshita_keyfile.json'

# Check if the file exists
>>>>>>> akshi_v1
if not os.path.exists(keyfile_path):
    raise FileNotFoundError(f"The file '{keyfile_path}' does not exist. Please check the path.")

# Set the environment variable to point to the service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path
<<<<<<< HEAD
os.environ['MLFLOW_GCS_BUCKET'] = 'ml-flow-remote-tracker-bucket'

mlflow.set_tracking_uri("http://35.231.231.140:5000/")
mlflow.set_experiment("Brain-Tumor-Classification")



physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("TensorFlow with Metal enabled with {} GPUs".format(len(physical_devices)))
    except RuntimeError as e:
        print(e)


def preprocessing_for_training():
    path = './data/Training/'
    train_datagen = ImageDataGenerator(
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
    return train_generator

def preprocessing_for_testing_inference(batchSize):
    path = './data/Testing/'
    test_val_datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator = test_val_datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=batchSize,
        class_mode='categorical',
        shuffle=False
    )
    return test_generator


model = models.Sequential([
    layers.InputLayer(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='sigmoid')
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
=======


# os.environ['MLFLOW_TRACKING_URI'] = 'http://35.231.231.140:5000'  
os.environ['MLFLOW_GCS_BUCKET'] = 'ml-flow-remote-tracker-bucket'

mlflow.set_tracking_uri("http://35.231.231.140:5000/")
mlflow.set_experiment("test")


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Start an MLflow run
with mlflow.start_run() as run:
    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Log a parameter
    mlflow.log_param("n_estimators", 100)

    # Log a metric
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Get the run ID
    run_id = run.info.run_id

print("complete")
# Initialize the client with explicit credentials
# credentials = service_account.Credentials.from_service_account_file(keyfile_path)
# client = storage.Client(credentials=credentials)

# # Verify the model upload
# bucket_name = os.environ.get('MLFLOW_GCS_BUCKET')
# if not bucket_name:
#     raise ValueError("Environment variable 'MLFLOW_GCS_BUCKET' is not set.")

# bucket = client.bucket(bucket_name)

# # Check if the model directory exists in the GCP bucket
# prefix = f"0/{run_id}/artifacts/model"
# blobs = list(bucket.list_blobs(prefix=prefix))

# if blobs:
#     print(f"Model successfully uploaded to GCP bucket '{bucket_name}' under '{prefix}'.")
# else:
#     print(f"Model upload failed. No artifacts found in GCP bucket '{bucket_name}' under '{prefix}'.")
>>>>>>> akshi_v1
