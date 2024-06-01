#!/usr/bin/env bash

# Initialize the metastore
airflow db init

# Run the scheduler in the background
airflow scheduler &> /dev/null &

# Create user
airflow users create -u mlopsproject -p admin -r Admin -e admin@admin.com -f admin -l admin

# Run the web server in foreground (for docker logs)
exec airflow webserver
