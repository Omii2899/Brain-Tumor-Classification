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
if not os.path.exists(keyfile_path):
    raise FileNotFoundError(f"The file '{keyfile_path}' does not exist. Please check the path.")

# Set the environment variable to point to the service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = keyfile_path


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