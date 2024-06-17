from airflow import DAG
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import os
import sys
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom imports
import dags.utils.config as config

default_args = {
    "owner": 'airflow',
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

def get_next_batch_folder():
    last_folder = Variable.get("last_processed_batch", default_var="batch-1")
    return last_folder

def set_next_batch_folder():
    last_folder = Variable.get("last_processed_batch", default_var="batch-1")
    folder_number = int(last_folder.strip('batch-'))
    next_folder = f"batch-{folder_number + 1}"
    Variable.set("last_processed_batch", next_folder)

with DAG(
    dag_id = "batch_train_model_data_and_store",
    description = "This DAG is responsible for training in batches",
    start_date =datetime(2024,5,15,2),
    schedule_interval = None,
    default_args=default_args,
    catchup = False,
    template_searchpath=["/opt/airflow/dags/utils"]
) as dag:
    task_get_batch_number_to_process = PythonOperator(
        task_id = "get_batch_number_to_process",
        python_callable = get_next_batch_folder
    )

    task_batch_gcs_psv_to_gcs_csv = BigQueryInsertJobOperator(
    task_id="merge_batch_data_from_psv_to_csv",
    gcp_conn_id=config.GCP_CONN_ID,
    configuration={
        "query": {
            "query": "{% include '/merge_data_from_psv_to_csv_batch.sql' %}",
            "useLegacySql": False
        }
    }
    )

    task_set_batch_number_to_process = PythonOperator(
        task_id = "set_batch_number_to_process",
        python_callable = set_next_batch_folder
    )

    task_get_batch_number_to_process >> task_batch_gcs_psv_to_gcs_csv >> task_set_batch_number_to_process