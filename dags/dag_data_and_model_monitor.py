from airflow import DAG
from datetime import datetime, timedelta
import os
import sys
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import BranchPythonOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom imports
import dags.utils.config as config
from dags.utils.helper import clean_pickle_files,  prepare_email_content
from dags.utils.data_preprocessing import data_preprocess_pipeline 
from dags.utils.data_split_utils import train_test_split
from dags.utils.data_scale_utils import scale_train_data, scale_test_data
from dags.utils.log_config import setup_logging
from dags.utils.schema_stats_utils import schema_stats_gen, schema_and_stats_validation
from dags.include.factory_data_processing import data_processing_task_group


DATA_DIR = config.DATA_DIR
STATS_SCHEMA_FILE = config.STATS_SCHEMA_FILE

logger = setup_logging(config.PROJECT_ROOT, "dag_data_and_model_monitor.py")


default_args = {
    "owner": 'airflow',
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

# Log the GCP bucket being used
BUCKET = config.bucket

#Tasks 

# Pull Predict.csv from GCS
# Pull Stat.JSON file from GCS
# Compare the stats
# If comparison fails send email else predict and then generate the results to be shown by GCS


# def branch_logic_schema_generation():
#     """
#     Determines the next task in a workflow based on the existence of a schema and stats file in a GCS bucket.

#     Args:
#         None

#     Returns:
#         str: The name of the next task to execute, either 'validate_data_schema_and_stats' or 'generate_schema_and_stats'.
#     """
#     hook = GCSHook(gcp_conn_id=config.GCP_CONN_ID)
#     file_exists = hook.exists(bucket_name=config.bucket, object_name='artifacts/schema_and_stats.json')

#     if file_exists:
#         return 'if_validate_data_schema_and_stats'
#     else:
#         return 'generate_schema_and_stats'


def create_directory():
    os.makedirs("data/artifact", exist_ok=True)

with DAG(
    dag_id = "monitor_data_and_model",
    description = "This DAG is responsible for data and model monitoring",
    start_date =datetime(2024,5,15,2),
    schedule_interval = None,
    default_args=default_args,
    catchup = False,
    template_searchpath=["/opt/airflow/dags/utils"]
) as dag:

    create_directory_task = PythonOperator(
        task_id="create_local_directory",
        python_callable=create_directory,
    )
    
    task_pull_predict_data = GCSToLocalFilesystemOperator(
        task_id="download_production_data",
        object_name="data/modified_data/prod_data/predict.csv",
        bucket=BUCKET,
        filename="data/predict.csv",
    )

    task_pull_schema_validation = GCSToLocalFilesystemOperator(
        task_id="download_schema_validation_data",
        object_name=f"artifact/{STATS_SCHEMA_FILE}",
        bucket=BUCKET,
        filename="data/artifact/schema_and_stats.json",
    )

    task_dummy_end = DummyOperator(task_id='end_task')

    create_directory_task >> task_pull_predict_data >> task_pull_schema_validation >> task_dummy_end

    # task_data_schema_and_statastics_validation = BranchPythonOperator(
    #     task_id='if_validate_data_schema_and_stats',
    #     python_callable=schema_and_stats_validation,
    #     trigger_rule='none_failed'
    # )

    # task_prepare_email_validation_failed = PythonOperator(
    #     task_id='prepare_email_content',
    #     python_callable=prepare_email_content,
    #     provide_context=True,
    # )

    # task_send_email_validation_failed = EmailOperator(
    #     task_id='email_validation_failed',
    #     to='derilraju@gmail.com',
    #     subject='Airflow Alert - Initial Train Pipeline',
    #     html_content="{{ task_instance.xcom_pull(task_ids='prepare_email_content') }}"
    # )

    # task_trigger_modelling_dag = TriggerDagRunOperator(
    #     task_id="trigger_modelling_dag",
    #     trigger_dag_id="model_data_and_store",
    # )

    # task_gcs_psv_to_gcs_csv >> task_if_schema_generation_required
    # task_if_schema_generation_required >> task_data_schema_and_statastics_validation
    # task_if_schema_generation_required >> task_schema_and_statastics_generation >> task_push_generated_schema_data >> task_data_schema_and_statastics_validation

    # task_data_schema_and_statastics_validation >> task_prepare_email_validation_failed >> task_send_email_validation_failed
    # # task_data_schema_and_statastics_validation >> task_train_test_split >> [task_X_train_data_preprocessing, task_X_test_data_preprocessing] >> task_scale_train_data >> task_scale_test_data >> [task_push_scaler, task_push_X_train_data, task_push_X_test_data, task_push_y_train_data, task_push_y_test_data] >> task_cleanup_files >> task_trigger_modelling_dag
    # task_data_schema_and_statastics_validation >> data_processing_task_group(dag, config.DATA_DIR) >> task_trigger_modelling_dag