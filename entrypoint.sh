#!/usr/bin/env bash
# gcloud auth activate-service-account aadarsh-727@tensile-topic-424308-d9.iam.gserviceaccount.com --key-file=tensile-topic-424308-d9-7418db5a1c90.json 
# #####
# gcloud config set project tensile-topic-424308-d9
# Initialize the metastore
airflow db init

# Run the scheduler in the background
airflow scheduler &> /dev/null &

# Create user
airflow users create -u mlopsproject -p admin -r Admin -e admin@admin.com -f admin -l admin

# Run the web server in foreground (for docker logs)
exec airflow webserver

# Run the jupyter notebook
# jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root