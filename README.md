# Brain-Tumor-Classification

This repository contains the code and configuration files for a Brain Tumor Detection MLOps project. The project includes data versioning, data pipelines, and Docker for containerization.

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

```
git clone https://github.com/Omii2899/Brain-Tumor-Classification.git
```
```
cd Brain-Tumor-Detection
```

### 2. Create a Python Virtual Environment
Create a virtual environment to manage project dependencies:
```
pip install virtualenv
```
```
python -m venv <virtual_environment_name>
```
```
source <virtual_environment_name>/bin/activate 
```

### 3. Install the Dependencies
Install the necessary dependencies using the requirements.txt file:
```
pip install -r requirements.txt
```

### 4. Get the Data from Remote Source
Pull the data from the remote source using DVC:

```
dvc pull
```

### Description of Files and Folders
#### Project Structure:
```
├── DVC
│   ├── config
│   ├── .gitignore
├── Data
│   ├── testing
│   │   ├── ...
│   ├── training
│   │   ├── ...
├── Dag
│   ├── datapipeline.py
│   └── scripts
│       ├── logger.py
│       ├── preprocessing.py
│       └── statistics.py
├── .dvcignore
├── .gitignore
├── data.dvc
├── Dockerfile
├── entrypoint.sh
├── requirements.txt
```

#### Source code files

<ol>
    <li><strong>Data Version Control (DVC)</strong>
        <ul>
            <li><strong>config</strong>: DVC configuration file for setting up data versioning.</li>
            <li><strong>.gitignore</strong>: Specifies which DVC files should be ignored by Git.</li>
        </ul>
    </li>
    <br>
    <li><strong>Data</strong>
        <ul>
            <li><strong>Testing</strong>: Contains subfolders with testing images.</li>
            <li><strong>Training</strong>: Contains subfolders with training images.</li>
        </ul>
    </li>
    <br>
    <li><strong>Source/DAG </strong>
        <ul>
            <li><strong>datapipeline.py</strong>: Orchestrates the data pipeline process, integrating various preprocessing and processing scripts.</li>
            <li><strong>scripts</strong>: 
                <ul>
                    <li><strong>logger.py</strong>: Sets up and configures logging for the project. It creates a logger instance with a specific format and log level.</li>
                    <li><strong>preprocessing.py</strong>: Handles image preprocessing for both training and testing phases using TensorFlow's <code>ImageDataGenerator</code>. It contains two primary functions: <code>preprocessing_for_training </code> that applies various augmentation techniques, such as normalization, rotation, width and height shifts, shearing, zooming, and horizontal flipping, to enhance the training datasetand and <code>preprocessing_for_testing_inference</code> for normalizing the data.</li>
                    <li><strong>statistics.py</strong>: Captures statistics of each image in each class, generates histograms, and validates images against these histograms using OpenCV.</li>
                </ul>
            </li>
        </ul>
    </li>
    <br>
    <li><strong>Configuration and Ignore Files</strong>
        <ul>
            <li><strong>.dvcignore</strong>: Specifies files and directories that DVC should ignore.</li>
            <li><strong>.gitignore</strong>: Specifies files and directories that Git should ignore.</li>
            <li><strong>data.dvc</strong>: DVC data tracking file.</li>
        </ul>
    </li>
    <br>
    <li><strong>Docker Configuration</strong>
        <ul>
            <li><strong>Dockerfile</strong>: Contains instructions to build a Docker image for the project.</li>
        </ul>
    </li>
    <br>
    <li><strong>Entry Point</strong>
        <ul>
            <li><strong>entrypoint.sh</strong>: Shell script to set up the environment and run the application inside Docker.</li>
        </ul>
    </li><br>
    <li><strong>Dependencies</strong>
        <ul>
            <li><strong>requirements.txt</strong>: Lists the Python dependencies needed for the project.</li>
        </ul>
    </li>
</ol>



