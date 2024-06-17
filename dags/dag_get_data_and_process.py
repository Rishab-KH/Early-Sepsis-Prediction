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
from dags.utils.helper import clean_pickle_files
from dags.utils.data_preprocessing import data_preprocess_pipeline 
from dags.utils.data_split_utils import train_test_split
from dags.utils.data_scale_utils import scale_train_data, scale_test_data
from dags.utils.log_config import setup_logging
from dags.utils.schema_stats_utils import schema_stats_gen, schema_and_stats_validation


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

def prepare_email_content(**context):
    ti = context['ti']
    validation_message = ti.xcom_pull(task_ids='if_validate_data_schema_and_stats', key='validation_message')
    
    dag_run = context['dag_run']
    dag_id = dag_run.dag_id
    execution_date = dag_run.execution_date.isoformat()
    task_id = ti.task_id
    owner = ti.task.dag.owner
    
    # Constructing the HTML content for the email.
    html_content = f"""
    <h3>Validation of Schema/Stats Failed</h3>
    <p>Find the error below:</p>
    <p>{validation_message}</p>
    <br>
    <strong>DAG Details:</strong>
    <ul>
        <li>DAG ID: {dag_id}</li>
        <li>Task ID: {task_id}</li>
        <li>Execution Date: {str(execution_date)}</li>
        <li>Owner: {owner}</li>
    </ul>
    <p>This is an automated message from Airflow. Please do not reply directly to this email.</p>
    """
    return html_content


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

    task_get_data_directory = PythonOperator(
        task_id = "get_data_location",
        python_callable=lambda: config.DATA_DIR
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
        subject='Airflow Alert',
        html_content="{{ task_instance.xcom_pull(task_ids='prepare_email_content') }}"
    )

    task_train_test_split = PythonOperator(
        task_id='train_test_split',
        python_callable=train_test_split,
        trigger_rule='none_failed'
    )

    task_X_train_data_preprocessing = PythonOperator(
        task_id='preprocess_X_train',
        python_callable=data_preprocess_pipeline,
        op_kwargs={'data_input': 'X_train.pkl', 'target_input': 'y_train.pkl', 'data_output':'X_train_processed.pkl'}
    )

    task_X_test_data_preprocessing = PythonOperator(
        task_id='preprocess_X_test',
        python_callable=data_preprocess_pipeline,
        op_kwargs={'data_input': 'X_test.pkl', 'target_input': 'y_test.pkl', 'data_output':'X_test_processed.pkl'}
    )

    task_scale_train_data = PythonOperator(
        task_id='scale_train_data',
        python_callable=scale_train_data,
    )

    task_scale_test_data = PythonOperator(
        task_id='scale_test_data',
        python_callable=scale_test_data,
    )

    task_push_scaler = LocalFilesystemToGCSOperator(
       task_id="push_scaler_to_gcs",
       gcp_conn_id=config.GCP_CONN_ID,
       src="scaler.pkl",
       dst="artifacts/scaler.pkl",
       bucket=BUCKET
    )

    task_push_X_train_data = LocalFilesystemToGCSOperator(
       task_id="push_X_train_data_to_gcs",
       gcp_conn_id=config.GCP_CONN_ID,
       src="X_train_processed_scaled.pkl",
       dst="data/processed_data/X_train.pkl",
       bucket=BUCKET
    )

    task_push_X_test_data = LocalFilesystemToGCSOperator(
       task_id="push_X_test_data_to_gcs",
       gcp_conn_id=config.GCP_CONN_ID,
       src="X_test_processed_scaled.pkl",
       dst="data/processed_data/X_test.pkl",
       bucket=BUCKET
    )

    task_push_y_train_data = LocalFilesystemToGCSOperator(
       task_id="push_y_train_data_to_gcs",
       gcp_conn_id=config.GCP_CONN_ID,
       src="y_train.pkl",
       dst="data/processed_data/y_train.pkl",
       bucket=BUCKET
    )

    task_push_y_test_data = LocalFilesystemToGCSOperator(
       task_id="push_y_test_data_to_gcs",
       gcp_conn_id=config.GCP_CONN_ID,
       src="y_test.pkl",
       dst="data/processed_data/y_test.pkl",
       bucket=BUCKET
    )

    task_cleanup_files = PythonOperator(
        task_id="clean_pickle_files",
        python_callable=clean_pickle_files,
        op_kwargs={"directory": os.getcwd()}
    )

    task_trigger_modelling_dag = TriggerDagRunOperator(
        task_id="trigger_modelling_dag",
        trigger_dag_id="model_data_and_store",
    )

    task_gcs_psv_to_gcs_csv >> task_get_data_directory >> task_if_schema_generation_required
    task_if_schema_generation_required >> task_data_schema_and_statastics_validation
    task_if_schema_generation_required >> task_schema_and_statastics_generation >> task_push_generated_schema_data >> task_data_schema_and_statastics_validation

    task_data_schema_and_statastics_validation >> task_prepare_email_validation_failed >> task_send_email_validation_failed
    task_data_schema_and_statastics_validation >> task_train_test_split >> [task_X_train_data_preprocessing, task_X_test_data_preprocessing] >> task_scale_train_data >> task_scale_test_data >> [task_push_scaler, task_push_X_train_data, task_push_X_test_data, task_push_y_train_data, task_push_y_test_data] >> task_cleanup_files >> task_trigger_modelling_dag