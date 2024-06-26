# Brain-Tumor-Classification

This project is designed to develop, deploy, and maintain a machine learning model for brain tumor classification. The project utilizes a Machine Learning Operations (MLOps) approach to streamline the development, deployment, and monitoring of the model. The project directory is structured to support data version control, modular coding, and containerized deployment.

## Introduction

Brain tumors are a significant health challenge, with approximately 24,810 adults in the United States diagnosed in 2023. The complexity and variability of brain tumors make accurate diagnosis difficult, especially in regions lacking skilled medical professionals. This project leverages machine learning to develop an end-to-end ML pipeline for automated brain tumor detection, aiming to provide scalable, reliable, and timely diagnostic support.

## Dataset Information

The dataset combines MRI images from three sources: figshare, SARTAJ, and Br35H. It includes 7023 JPEG images of human brains, categorized into four classes: glioma, meningioma, no tumor, and pituitary.

- **Dataset Name**: Brain Tumor MRI Images
- **Size**: 7023 images
- **Format**: JPEG
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary
- **Sources**:
  - [figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
  - [SARTAJ](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
  - [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection?select=no)

All data used are sourced from publicly available datasets with proper usage permissions.

## Project Workflow

![picture alt](assets/model-architecture.png)

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- [Git](https://www.git-scm.com/downloads)
- [Docker](https://www.docker.com/get-started/)
- [Airflow](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
- [DVC](https://airflow.apache.org/docs/apache-airflow/stable/start.html) (Data Version Control)
- [Python](https://www.python.org/downloads/) 3.x
- Pip (Python package installer)
- Google Cloud Platform (GCP) Account

## Getting Started

To get started with the project, follow these steps:

### 1. Clone the Repository

Clone the repository using the following command:

```sh
git clone https://github.com/Omii2899/Brain-Tumor-Classification.git
cd Brain-Tumor-Detection
```

### 2. Create a Python Virtual Environment
Create a virtual environment to manage project dependencies:

```sh
pip install virtualenv
python -m venv <virtual_environment_name>
source <virtual_environment_name>/bin/activate 
```

### 3. Install the Dependencies
Install the necessary dependencies using the requirements.txt file:

```sh
pip install -r requirements.txt
```

### 4. Get the Data from Remote Source
Pull the data from the remote source using DVC:

```sh
dvc pull
``` 

### 5. Add the Key File:
You need to add the key file in src/keys folder. For security purposes, we have not included this file. To obtain this file, please contact [Aadarsh](mailto:siddha.a@northeastern.edu)  

## Description of Files and Folders
#### Project Structure:
```plaintext
├── .dvc
│   ├── config
│   ├── .gitignore
├── data
│   ├── Testing
│   │   ├── ...
│   ├── Training
│   │   ├── ...
├── frontend
│   ├── app.py
│   ├── dockerfile
│   ├── requirements.txt
│   ├── kubernetes
│       ├── deployment.yaml
│       ├── namespace.yaml
│       ├── service.yaml
├── backend
├── src
│   ├── dags
│   │   ├── scripts
│   │       ├── logger.py
│   │       ├── preprocessing.py
│   │       ├── statistics.py
│   │   ├── datapipeline.py
│   └── keys
│       ├── keyfile.json
├── .dvcignore
├── .gitignore
├── data.dvc
├── dockerfile
├── entrypoint.sh
├── requirements.txt
```

#### Source Code Files:

**The below files are in the `src` folder**

1. **Data Pipeline**

   - `datapipeline.py`: Orchestrates the entire data processing workflow, including data ingestion, preprocessing, and feature engineering.

2. **Logging Configuration**

   - `logger.py`: Configures the logging system for the project, defining log formats, levels, and handlers to ensure proper tracking and debugging of processes.

3. **Initial Preprocessing**

   - `preprocessing.py`: Performs initial preprocessing tasks on the dataset, such as cleaning, normalization, and transformation of raw data into a suitable format for further analysis and model training.

4. **Statistical Analysis**

   - `statistics.py`: Conducts statistical analysis on the dataset, calculating various descriptive statistics and generating insights about the data distribution and relationships between features.

![picture alt](assets/metrics.jpg)
   
5. **Machine Learning Experiment Tracking**

   - `example-mlflow.ipynb`: A Jupyter notebook demonstrating the use of MLflow for tracking machine learning experiments, including logging parameters, metrics, and model artifacts.

## Data Card After Preprocessing and Feature Engineering

| Variable Name            | Role    | DType    | Description                                                                        |
|--------------------------|---------|----------|------------------------------------------------------------------------------------|
| Image_ID                 | ID      | int64    | Unique identifier for each MRI image                                               |
| Acquisition_Date         | ID      | datetime | Date when the MRI image was acquired                                               |
| Tumor_Type               | Target  | int64    | Category of tumor: 0 - No Tumor, 1 - Glioma, 2 - Meningioma, 3 - Pituitary         |
| Image_Pixels             | Feature | ndarray  | Pixel values of the MRI image                                                      |
| Image_Resolution         | Feature | int64    | Resolution of the MRI image                                                        |
| Image_Width              | Feature | int64    | Width of the MRI image in pixels                                                   |
| Image_Height             | Feature | int64    | Height of the MRI image in pixels                                                  |
| Pixel_Spacing            | Feature | float64  | Spacing between pixels in the MRI image                                            |
| Augmented                | Feature | bool     | Whether the image was augmented (True/False)                                       |
| Augmentation_Type        | Feature | int64    | Type of augmentation applied (if any): 0 - None, 1 - Rotation, 2 - Flipping, etc.  |
| Brain_Region             | Feature | int64    | Region of the brain shown in the MRI image                                         |
| MRI_Machine_Type         | Feature | int64    | Type of MRI machine used for imaging                                               |
| Image_Contrast           | Feature | float64  | Contrast level of the MRI image                                                    |
| Noise_Level              | Feature | float64  | Noise level in the MRI image                                                       |
| Preprocessing_Steps      | Feature | object   | List of preprocessing steps applied to the image                                   |
| Is_Corrupted             | Feature | bool     | Whether the image is corrupted (True/False)                                        |
| Tumor_Size               | Feature | float64  | Size of the tumor detected in the MRI image (if any)                               |
| Diagnosis_Confirmed      | Feature | bool     | Whether the diagnosis has been confirmed by a professional (True/False)            |
| Diagnosis_Date           | Feature | datetime | Date when the diagnosis was confirmed                                              |

### Description

This data card provides an overview of the variables present in the dataset after preprocessing and feature engineering. Each variable has a specific role, data type, and description to help understand its significance in the context of brain tumor detection and classification using MRI images. This comprehensive data card can be included in the README file to provide clarity on the dataset's structure and the preprocessing steps applied. This dataset only includes image-related information and does not contain any personal information about patients.

## Model Train and Inference

1. **Training and Inference**

   - `build.py`: Initializes the Vertex AI platform, trains the model, and saves it to a bucket.
   - `inference.py`: Utilizes the predict function for inference.

2. **Docker Image Creation for Training and Serving**

   - Setup Docker, create train and serve docker images, push to Artifact Repository:
     ```sh
     gcloud auth configure-docker us-central1-docker.pkg.dev
     ```

     **File paths**: `src/trainer/Dockerfile` and `src/serve/Dockerfile`

     **Commands**:
     ```sh
     docker buildx build --platform linux/amd64 -f trainer/Dockerfile -t us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/trainer:v1 . --load
     docker push us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/trainer:v1

     docker buildx build --platform linux/amd64 -f serve/Dockerfile -t us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/serve:v1 . --load
     docker push us-east1-docker.pkg.dev/[YOUR_PROJECT_ID]/[FOLDER_NAME]/

serve:v1
     ```

## Model Versioning

- Model Registry of Vertex AI does the model versioning.
- Artifact Registry versions the docker images.

## Running the data pipeline

To run the pipeline, you can use Docker for containerization.

1. Build the Docker Image
```sh
docker build -t image-name:tag-name .
```
2. Verify the image 
```
docker images
```

3. Run the built image
```sh
docker run -it --rm -p 8080:8080 image-name:tag-name
```

The application should now be running and accessible at [http://localhost:8080](http://localhost:8080).

Use the below credentials:
- **User**: mlopsproject
- **Password**: admin

*Note: If the commands fail to execute, ensure that virtualization is enabled in your BIOS settings. Additionally, if you encounter permission-related issues, try executing the commands by prefixing them with `sudo`.*

4. Trigger the Airflow UI
```sh
python src/dags/datapipeline.py
```

## Data storage and Model Registry:

### 1. Storage buckets
![picture alt](assets/GCP-buckets.jpg)

### 2. Data buckets
![picture alt](assets/data-bucket.png)

- **/data**: This directory contains the dataset used for training and testing the ML model.
- **InferenceLogs/**: This directory is dedicated to storing inference logs, facilitating model evaluation and improvement:
  - **ImageLogs/**: Subfolder for storing user input images along with correct predictions made by the model. These logs are valuable for validating model accuracy.
  - **ImageLogsWithFeedback/**: Subfolder for storing user input images that were incorrectly predicted by the model, categorized by the label provided by the user. This data is essential for retraining and enhancing the model's performance.

### DAG:

#### 1. Data and model build pipeline
![picture alt](assets/data-pipeline.jpg)

1. **check_source**: Checking the data source to verify its availability.
2. **download_data**: Downloading the necessary data if the source is valid.
3. **capture_statistics**: Captures statistics about the data, such as summary statistics, distributions, and other relevant metrics.
4. **augment_input_data**: Performing data augmentation, feature engineering, and other preprocessing steps.
5. **transform_testing_data**: Transforming the testing data to ensure it is in the correct format for model evaluation.
6. **building_model**: Builds the machine learning model using the prepared data.
7. **send_email**: Sends an email notification upon a successful model build.

#### 2. Model Retraining pipeline
![picture alt](assets/retrain-pipeline.jpg)

1. **check_source_flag**: Checks if there are more than 50 wrongly predicted images in the bucket.
2. **flag_false**: Ends the process if there are 50 or fewer wrongly predicted images.
3. **flag_true**: Proceeds to model retraining if there are more than 50 wrongly predicted images.
4. **retrain_model**: Initiates the re-training of the model with updated data.
5. **send_email**: Sends an email notification once model retraining is completed.

## Application Interface

![picture alt](assets/ui-1.png)


## **Disclaimer**

**Please note that any images you upload will be stored with us. By uploading an image, you consent to its storage and use for the purposes of improving our brain tumor classification model. We are committed to ensuring the privacy and security of your data and will not share it with any third parties without your explicit consent.**


## Contributors

[Aadrash Siddha](https://github.com/Omii2899)  
[Akshita Singh](https://github.com/akshita-singh-2000)  
[Praneith Ranganath](https://github.com/Praneith)  
[Shaun Kirtan](https://github.com/)  
[Yashasvi Sharma](https://github.com/yashasvi14)
```git 