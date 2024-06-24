from airflow import DAG
from datetime import datetime, timedelta
import os
import pandas as pd
import sys
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom imports
import dags.utils.config as config
from dags.utils.helper import prepare_email_content_prod
from dags.utils.log_config import setup_logging
from dags.utils.schema_stats_utils import schema_and_stats_validation




# Stats Schema File
STATS_SCHEMA_FILE = config.STATS_SCHEMA_FILE

logger = setup_logging(config.PROJECT_ROOT, "dag_data_and_model_monitor.py")


default_args = {
    "owner": 'airflow',
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

# Log the GCP bucket being used
BUCKET = config.bucket


def get_data_location():
    """Return the data directory"""
    return config.PREDICT_DIR

def drop_created_at_column(**kwargs):
    """Function to drop the created_at column from the DataFrame"""
    ti = kwargs['ti']
    df_path = ti.xcom_pull(task_ids='get_data_location')
    df = pd.read_csv(df_path)
    if 'created_at' in df.columns:
        df.drop(columns=['created_at'], inplace=True)
        df.to_csv(df_path, index=False)
    logger.info("Dropped the 'created_at' column from the DataFrame")
    logger.info(f"Existing columns in production data {df.columns}")
    
with DAG(
    dag_id = "monitor_data_and_model",
    description = "This DAG is responsible for data and model monitoring",
    start_date =datetime(2024,5,15,2),
    schedule_interval="@weekly",
    default_args=default_args,
    catchup = False,
    template_searchpath=["/opt/airflow/dags/utils"]
) as dag:


    task_get_data_directory = PythonOperator(
        task_id = "get_data_location",
        python_callable=get_data_location
    )

    task_drop_created_at_column = PythonOperator(
        task_id='prod_data_preprocess',
        python_callable=drop_created_at_column,
        provide_context=True
    )

    task_data_schema_and_statastics_validation = BranchPythonOperator(
        task_id='if_validate_data_schema_and_stats',
        python_callable=schema_and_stats_validation
    )

    task_prepare_email_validation_failed = PythonOperator(
        task_id='prepare_email_content',
        python_callable=prepare_email_content_prod,
        provide_context=True,
    )

    task_send_email_validation_failed = EmailOperator(
        task_id='email_validation_failed',
        to='derilraju@gmail.com',
        subject='Airflow Alert - Batch Retrain Pipeline',
        html_content="{{ task_instance.xcom_pull(task_ids='prepare_email_content') }}"
    )

    # Dummy operator to signify the end of parallel tasks
    task_end = DummyOperator(
        task_id="end_monitor_task"
    )
    

    task_get_data_directory >> task_drop_created_at_column >> task_data_schema_and_statastics_validation 

    # If validation succeeds return task_end from utils/schema_stats_utils 
    task_data_schema_and_statastics_validation >> task_end
    # If validation fails return email validation fail which will create the email content and send it to admin for monitoring
    task_data_schema_and_statastics_validation >> task_prepare_email_validation_failed >> task_send_email_validation_failed
