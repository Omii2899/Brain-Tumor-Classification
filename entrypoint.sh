#!/usr/bin/env bash

# Initialize the metastore
airflow db init

# Disable example DAGs
sed -i 's/load_examples = True/load_examples = False/' ${AIRFLOW_HOME}/airflow.cfg

# Update airflow.cfg with Gmail SMTP settings
if ! grep -q "\[smtp\]" ${AIRFLOW_HOME}/airflow.cfg; then
    echo -e "\n[smtp]\nsmtp_host=smtp.gmail.com\nsmtp_starttls=True\nsmtp_ssl=False\nsmtp_user=yash141010@gmail.com\nsmtp_password=rllqeajmajoeuthj\nsmtp_port=587\nsmtp_mail_from=yash141010@gmail.com\nsmtp_timeout=30\nsmtp_retry_limit=5" >> ${AIRFLOW_HOME}/airflow.cfg
else
    sed -i "s/^smtp_host.*/smtp_host=smtp.gmail.com/" ${AIRFLOW_HOME}/airflow.cfg
    sed -i "s/^smtp_starttls.*/smtp_starttls=True/" ${AIRFLOW_HOME}/airflow.cfg
    sed -i "s/^smtp_ssl.*/smtp_ssl=False/" ${AIRFLOW_HOME}/airflow.cfg
    sed -i "s/^# smtp_user.*/smtp_user=yash141010@gmail.com/" ${AIRFLOW_HOME}/airflow.cfg
    sed -i "s/^# smtp_password.*/smtp_password=rllqeajmajoeuthj/" ${AIRFLOW_HOME}/airflow.cfg
    sed -i "s/^smtp_port.*/smtp_port=587/" ${AIRFLOW_HOME}/airflow.cfg
    sed -i "s/^smtp_mail_from.*/smtp_mail_from=yash141010@gmail.com/" ${AIRFLOW_HOME}/airflow.cfg
    sed -i "s/^smtp_timeout.*/smtp_timeout=30/" ${AIRFLOW_HOME}/airflow.cfg
    sed -i "s/^smtp_retry_limit.*/smtp_retry_limit=5/" ${AIRFLOW_HOME}/airflow.cfg
fi

# Run the scheduler in the background
airflow scheduler &> /dev/null &

# Create user (if not already created)
airflow users create -u mlopsproject -p admin -r Admin -e admin@admin.com -f admin -l admin || true

# Run the web server in foreground (for Docker logs)
exec airflow webserver