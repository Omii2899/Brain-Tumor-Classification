FROM python:3.8
LABEL maintainer="Aadarsh"

ARG AIRFLOW_VERSION=2.0.2
ARG AIRFLOW_HOME=/opt/airflow

WORKDIR ${AIRFLOW_HOME}
ENV AIRFLOW_HOME=${AIRFLOW_HOME}

RUN apt-get update -yqq && \
    apt-get install -yqq --no-install-recommends \
    wget \
    curl \
    git \
    gcc \
    && apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt 

# Upgrade pip separately to catch any issues
RUN pip install --upgrade pip 

# Add airflow user separately to catch any issues
RUN useradd -ms /bin/bash -d ${AIRFLOW_HOME} airflow

# Install Apache Airflow separately to catch any issues
RUN pip install apache-airflow==${AIRFLOW_VERSION} 

# Install requirements separately to catch any issues
RUN pip install -r /requirements.txt 

COPY ./entrypoint.sh ${AIRFLOW_HOME}/entrypoint.sh
RUN chmod +x ${AIRFLOW_HOME}/entrypoint.sh
#copy dags folder 
COPY ./dags ${AIRFLOW_HOME}/dags

RUN chown -R airflow: ${AIRFLOW_HOME}
USER airflow

EXPOSE 8080

ENTRYPOINT [ "/opt/airflow/entrypoint.sh" ]
