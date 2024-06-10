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
Project Structure:
    <ul>
        <li><strong>DVC Folder</strong>: Contains DVC configuration and .gitignore file for managing data versioning.</li>
        <li><strong>Data Folder</strong>: Contains subfolders for training and testing images.</li>
        <li><strong>Dag Folder</strong>: Contains the <code>datapipeline.py</code> script for data preprocessing and pipeline management.</li>
        <li><strong>.dvcignore</strong>: Specifies files and directories for DVC to ignore.</li>
        <li><strong>.gitignore</strong>: Specifies files and directories for Git to ignore.</li>
        <li><strong>data.dvc</strong>: DVC data tracking file.</li>
        <li><strong>Dockerfile</strong>: Contains instructions to build a Docker image for the project.</li>
        <li><strong>entrypoint.sh</strong>: Shell script to set up the environment and run the application inside Docker.</li>
        <li><strong>requirements.txt</strong>: List of Python dependencies needed for the project.</li>
    </ul>


