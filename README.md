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


