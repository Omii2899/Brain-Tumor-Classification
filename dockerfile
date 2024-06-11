FROM python:3.8
LABEL maintainer="Aadarsh"

ARG AIRFLOW_VERSION=2.0.2
ARG AIRFLOW_HOME=/mnt/airflow

WORKDIR ${AIRFLOW_HOME}
ENV AIRFLOW_HOME=${AIRFLOW_HOME}

RUN apt-get update -yqq && \
    apt-get install -yqq --no-install-recommends \
    wget \
    curl \
    git \
    gcc \
    libhdf5-dev \   
    libleveldb-dev \ 
    && apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
# RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-479.0.0-linux-x86_64.tar.gz \
#     && tar -xf google-cloud-cli-479.0.0-linux-x86_64.tar.gz \
#     && ./google-cloud-sdk/install.sh 

# Add gcloud to PATH
# ENV PATH $PATH:/opt/airflow/google-cloud-sdk/bin

COPY ./requirements.txt /requirements.txt 
# COPY ./src/keys/tensile-topic-424308-d9-7418db5a1c90.json ${AIRFLOW_HOME}/tensile-topic-424308-d9-7418db5a1c90.json

# Upgrade pip separately to catch any issues
RUN pip install --upgrade pip && \
    useradd -ms /bin/bash -d ${AIRFLOW_HOME} airflow && \
    pip install apache-airflow==${AIRFLOW_VERSION} && \
    pip install -r /requirements.txt

COPY ./entrypoint.sh ${AIRFLOW_HOME}/entrypoint.sh
RUN chmod +x ${AIRFLOW_HOME}/entrypoint.sh

# Copy dags folder 
COPY ./src ${AIRFLOW_HOME}/

RUN chown -R airflow: ${AIRFLOW_HOME}
USER airflow

ENV GOOGLE_APPLICATION_CREDENTIALS=/mnt/airflow/keys/tensile-topic-424308-d9-7418db5a1c90.json
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
EXPOSE 8080
EXPOSE 8888

ENTRYPOINT [ "/mnt/airflow/entrypoint.sh" ]
