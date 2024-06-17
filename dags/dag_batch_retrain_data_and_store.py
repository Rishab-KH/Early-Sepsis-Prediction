from airflow import DAG
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import os
import sys
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.operators.email import EmailOperator

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom imports
import dags.utils.config as config
from dags.utils.helper import prepare_email_content
from dags.utils.schema_stats_utils import schema_and_stats_validation

default_args = {
    "owner": 'airflow',
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

def get_next_batch_folder():
    last_folder = Variable.get("last_processed_batch", default_var="batch-1")
    folder_number = int(last_folder.strip('batch-'))
    if folder_number > 3:
        raise Exception("Out of batches, Terminating")
    return last_folder

def set_next_batch_folder():
    last_folder = Variable.get("last_processed_batch", default_var="batch-1")
    folder_number = int(last_folder.strip('batch-'))
    next_folder = f"batch-{folder_number + 1}"

    Variable.set("last_processed_batch", next_folder)

def get_next_batch_gs_location(ti):
    current_batch = ti.xcom_pull('get_batch_number')
    return f"{config.gsutil_URL}/data/modified_data/{current_batch}/finalDataset-000000000000.csv"

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
        task_id = "get_batch_number",
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
        task_id = "set_batch_number",
        python_callable = set_next_batch_folder
    )

    task_get_data_directory = PythonOperator(
        task_id = "get_data_location",
        python_callable=get_next_batch_gs_location
    )

    task_data_schema_and_statastics_validation = BranchPythonOperator(
        task_id='if_validate_data_schema_and_stats',
        python_callable=schema_and_stats_validation
    )

    task_prepare_email_validation_failed = PythonOperator(
        task_id='prepare_email_content',
        python_callable=prepare_email_content,
        provide_context=True,
    )

    task_send_email_validation_failed = EmailOperator(
        task_id='email_validation_failed',
        to='derilraju@gmail.com',
        subject='Airflow Alert - Batch Retrain Pipeline',
        html_content="{{ task_instance.xcom_pull(task_ids='prepare_email_content') }}"
    )

    task_dummy_end = DummyOperator(task_id='end_task', trigger_rule='none_failed')

    task_get_batch_number_to_process >> task_batch_gcs_psv_to_gcs_csv >> task_set_batch_number_to_process >> task_get_data_directory >> task_data_schema_and_statastics_validation
    task_data_schema_and_statastics_validation >> task_prepare_email_validation_failed >> task_send_email_validation_failed
    task_data_schema_and_statastics_validation >> task_dummy_end