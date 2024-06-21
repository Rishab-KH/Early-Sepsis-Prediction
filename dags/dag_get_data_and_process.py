from airflow import DAG
from datetime import datetime, timedelta
import os
import sys
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import BranchPythonOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.operators.email import EmailOperator

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom imports
import dags.utils.config as config
from dags.utils.helper import prepare_email_content
from dags.utils.log_config import setup_logging
from dags.utils.schema_stats_utils import schema_stats_gen, schema_and_stats_validation
from dags.include.factory_data_processing import data_processing_task_group


DATA_DIR = config.DATA_DIR
STATS_SCHEMA_FILE = config.STATS_SCHEMA_FILE

logger = setup_logging(config.PROJECT_ROOT, "dag_get_data_and_preprocess.py")


default_args = {
    "owner": 'airflow',
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

# Log the GCP bucket being used
BUCKET = config.bucket

def branch_logic_schema_generation():
    """
    Determines the next task in a workflow based on the existence of a schema and stats file in a GCS bucket.

    Args:
        None

    Returns:
        str: The name of the next task to execute, either 'validate_data_schema_and_stats' or 'generate_schema_and_stats'.
    """
    hook = GCSHook(gcp_conn_id=config.GCP_CONN_ID)
    file_exists = hook.exists(bucket_name=config.bucket, object_name='artifacts/schema_and_stats.json')

    if file_exists:
        return 'if_validate_data_schema_and_stats'
    else:
        return 'generate_schema_and_stats'

with DAG(
    dag_id = "train_data_preprocess_with_gcp",
    description = "This DAG is responsible for cleaning and preprocessing the raw data into usable format",
    start_date =datetime(2024,5,15,2),
    schedule_interval = None,
    default_args=default_args,
    catchup = False,
    template_searchpath=["/opt/airflow/dags/utils"]
) as dag:

    task_gcs_psv_to_gcs_csv = BigQueryInsertJobOperator(
    task_id="merge_data_from_psv_to_csv",
    gcp_conn_id=config.GCP_CONN_ID,
    configuration={
        "query": {
            "query": "{% include '/merge_data_from_psv_to_csv.sql' %}",
            "useLegacySql": False
        }
    }
    )

    task_if_schema_generation_required = BranchPythonOperator(
    task_id='if_schema_exists',
    python_callable=branch_logic_schema_generation
    )

    task_schema_and_statastics_generation = PythonOperator(
        task_id='generate_schema_and_stats',
        python_callable=schema_stats_gen
    )

    task_push_generated_schema_data = LocalFilesystemToGCSOperator(
       task_id="push_schema_to_gcs",
       gcp_conn_id=config.GCP_CONN_ID,
       src=STATS_SCHEMA_FILE,
       dst=f"artifacts/{STATS_SCHEMA_FILE}",
       bucket=BUCKET
    )

    task_get_data_directory = PythonOperator(
        task_id = "get_data_location",
        python_callable=lambda: config.DATA_DIR
    )

    task_data_schema_and_statastics_validation = BranchPythonOperator(
        task_id='if_validate_data_schema_and_stats',
        python_callable=schema_and_stats_validation,
        trigger_rule='none_failed'
    )

    task_prepare_email_validation_failed = PythonOperator(
        task_id='prepare_email_content',
        python_callable=prepare_email_content,
        provide_context=True,
    )

    task_send_email_validation_failed = EmailOperator(
        task_id='email_validation_failed',
        to='derilraju@gmail.com',
        subject='Airflow Alert - Initial Train Pipeline',
        html_content="{{ task_instance.xcom_pull(task_ids='prepare_email_content') }}"
    )

    task_trigger_modelling_dag = TriggerDagRunOperator(
        task_id="trigger_modelling_dag",
        trigger_dag_id="model_data_and_store",
    )

    task_gcs_psv_to_gcs_csv >> task_get_data_directory >> task_if_schema_generation_required
    task_if_schema_generation_required >> task_data_schema_and_statastics_validation
    task_if_schema_generation_required >> task_schema_and_statastics_generation >> task_push_generated_schema_data >> task_data_schema_and_statastics_validation

    task_data_schema_and_statastics_validation >> task_prepare_email_validation_failed >> task_send_email_validation_failed
    task_data_schema_and_statastics_validation >> data_processing_task_group(dag, config.DATA_DIR, train_type="initial_train") >> task_trigger_modelling_dag