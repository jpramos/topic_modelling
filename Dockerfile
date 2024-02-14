FROM python:3.11-slim-buster

RUN set -ex \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends libpq-dev build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install required packages
# RUN pip install dbt-databricks
RUN pip install langchain langchain-openai faiss-cpu

# COPY ./dbt_env/ /root/.dbt/
# COPY . /project
# ENV DBT_PROFILES_DIR /dbt_env

WORKDIR /project
