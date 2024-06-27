from airflow import DAG
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import sys
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom imports
import utils.config as config
from utils.helper import prepare_email_content_schema_prod, prepare_email_content_statistics_prod
from utils.log_config import setup_logging

import gcsfs
import json


# Stats Schema Path
STATS_SCHEMA_PATH = config.STATS_SCHEMA_FILE_GCS

logger = setup_logging(config.PROJECT_ROOT, "dag_data_and_model_monitor.py")


default_args = {
    "owner": 'airflow',
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

# Log the GCP bucket being used
BUCKET = config.bucket


# Get data directory
def get_data_dir():
    """Return the data directory"""
    return config.PREDICT_DIR

def drop_created_at_column(**kwargs):
    """Function to modify to read the df from original src and then save a copy in data_dir
    Then drop the created_at column from the DataFrame"""

    ti = kwargs['ti']
    df_path = ti.xcom_pull(task_ids='get_data_directory')
    df_path_list = df_path.split("/")
    modified_df_path = "/".join(df_path_list[:-1]) + f"/archived_{df_path_list[-1]}" # Src of the data
    
    df = pd.read_csv(modified_df_path)
    
    if 'created_at' in df.columns:
        df.drop(columns=['created_at'], inplace=True)
        # A copy of the CSV without the created at column with name ProdDataset.csv
        df.to_csv(df_path, index=False) 
    logger.info("Dropped the 'created_at' column from the DataFrame")
    logger.info(f"Existing columns in production data {df.columns}")

def load_schema_and_stats(schema_file=STATS_SCHEMA_PATH):
    """
    Load the schema and statistics from a JSON file.

    Args:
        schema_file (str): Path to the schema and statistics file.

    Returns:
        dict: Loaded schema and statistics.
    """
    try:
        # with open(schema_file, 'r') as f:
        #     schema_and_stats = json.load(f)

        gcs_file_system = gcsfs.GCSFileSystem(project=config.GCP_PROJECT_NAME)
        gcs_json_path = config.STATS_SCHEMA_FILE_GCS
        with gcs_file_system.open(gcs_json_path) as f:
            schema_and_stats = json.load(f)
        logger.info(f"Schema and statistics loaded from {schema_file}.")
        return schema_and_stats
    except Exception as e:
        logger.error(f"Error loading schema and statistics from {schema_file}: {e}")
        raise

# Validate schema

def validate_schema(df):
    """
    Validate the schema of the DataFrame against the expected schema.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        schema (dict): Expected schema.

    Returns:
        bool: True if schema is valid, False otherwise.
        err_msg (str): Error message, if any else None
    """
    err_msg = None
    # Load schema and statistics
    schema_and_stats = load_schema_and_stats()
    logger.info(f"schema for comparision between training and serving data is loaded successfully.")
    schema = schema_and_stat['schema']
    
    flag = True
    for column, dtype in schema.items():
        if "SepsisLabel" !=  column:
            if column not in df.columns:
                err_msg = f"Missing column: {column}\n"
                logger.error(err_msg)
                flag = False
                
            if column == "Age":
                if df["Age"].dtype not in [np.int64, np.float64]:
                    err_msg = f"Invalid type for column Age. Expected int64 or float64, got something else"
                    logger.error(err_msg)
                    flag = False

                    
            elif str(df[column].dtype) != dtype:
                err_msg = f"Invalid type for column {column}: Expected {dtype}, got {df[column].dtype}"
                logger.error(err_msg)
                flag = False
            
    logger.info("Schema validation passed.")
    # return True, err_msg
    if flag == False:
        err_msg = f"Schema validation failed: {err_msg}"
        logger.error(err_msg)
        return flag, err_msg
    return flag, err_msg # err_msg will be set to None if flag is True

def schema_validation(ti):
    data_dir = ti.xcom_pull('get_data_directory')
    try:
        df=pd.read_csv(data_dir, sep=",")
        logger.info(f"Data loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"File not found: {config.PREDICT_DIR}. Error: {e}")
        raise ValueError("Failed to Load Data for Schema Validation. Stopping DAG execution.")
    validate_schema_result, validate_schema_msg = validate_schema(df)
    if validate_schema_result == False:
        err_msg = f"Schema validation failed: {validate_schema_msg}"
        logger.error(err_msg)
    else:
        err_msg = None
        logger.info(f"Schema validation succeeded")
    validate_schema_message = err_msg
    ti.xcom_push(key='validation_schema_message', value=validate_schema_message)
    if validate_schema_result:
            return 'if_validate_data_statistics' 
    return 'prepare_email_schema_content'

def validate_statistics(df):
    """
    Validate statistics of the DataFrame against expected statistics.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        stats (dict): Expected statistics.

    Returns:
        bool: True if statistics are valid, False otherwise.
        err_msg (str): Error message, if any else None
    """
    err_msg = None
    # Load schema and statistics
    schema_and_stats = load_schema_and_stats()
    logger.info(f"Statistics for comparision between training and serving data is loaded successfully.")
    stats = schema_and_stats['statistics']

    try:
        flag = True
        for col, stat in stats.items():
            if col not in df.columns:
                err_msg = f"Missing column: {col}"
                logger.error(err_msg)
                flag = False
            
            if col == 'Patient_ID':
                if df[col].isnull().any():
                    err_msg = "The 'patient_id' column cannot have null values."
                    logger.error(err_msg)
                    flag = False
                continue

            if 'min' in stat and 'max' in stat:
                if stat['min'] is not None and stat['max'] is not None:
                    if df[col].min() < stat['min']:
                        logger.warning(f"Column {col} min value anomaly: {df[col].min()} < {stat['min']}")
                    if df[col].max() > stat['max']:
                        logger.warning(f"Column {col} max value anomaly: {df[col].max()} > {stat['max']}")

            if 'mean' in stat and 'std' in stat:
                if stat['mean'] is not None and stat['std'] is not None:
                    if not df[col].isnull().all():  # Check if any non-null values exist
                        if abs(df[col].mean() - stat['mean']) > 3 * stat['std']:
                            logger.warning(f"Column {col} mean value anomaly: {df[col].mean()} != {stat['mean']}")

            if 'median' in stat and 'std' in stat:
                if stat['median'] is not None and stat['std'] is not None:
                    if not df[col].isnull().all():  # Check if any non-null values exist
                        if abs(df[col].median() - stat['median']) > 3 * stat['std']:
                            logger.warning(f"Column {col} median value anomaly: {df[col].median()} != {stat['median']}")

            if 'null_count' in stat:
                null_count = df[col].isnull().sum()
                if stat['null_count'] is not None:
                    if null_count > stat['null_count']:
                        logger.warning(f"Column {col} null value count anomaly: {null_count} > {stat['null_count']}")

            if 'unique_values' in stat:
                if stat['unique_values'] is not None:
                    unique_values = df[col].unique()
                    if set(unique_values) != set(stat['unique_values']):
                        logger.warning(f"Column {col} unique values anomaly: {unique_values} != {stat['unique_values']}")

        logger.info("Statistical validation passed.")
        return flag, err_msg
    except Exception as e:
        err_msg = f"Error during statistical validation: {e}"
        logger.error(err_msg)
        flag = False
        return flag, err_msg




def stats_validation(ti):
    data_dir = ti.xcom_pull('get_data_directory')
    try:
        df=pd.read_csv(data_dir, sep=",")
        logger.info(f"Data loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"File not found: {config.PREDICT_DIR}. Error: {e}")
        raise ValueError("Failed to Load Data for Statstics Validation. Stopping DAG execution.")
    validate_stats_result, validate_stats_msg = validate_statistics(df)
    if validate_stats_result == False:
        err_msg = f"Stats validation failed: {validate_stats_msg}"
        logger.error(err_msg)
    else:
        err_msg = None
        logger.info(f"Statistics validation succeeded")
    validate_stats_message = err_msg
    ti.xcom_push(key='validation_email_message', value=validate_stats_message)
    if validate_stats_result:
            return 'end_monitor_task'
    return 'prepare_email_stats_content'


    
with DAG(
    dag_id = "monitor_data_and_model",
    description = "This DAG is responsible for data and model monitoring",
    start_date =datetime(2024,5,15,2),
    schedule_interval="@weekly",
    default_args=default_args,
    catchup = False,
    template_searchpath=["/opt/airflow/dags/utils","/home/airflow/gcs/dags/utils"]
) as dag:


    task_get_data_directory = PythonOperator(
        task_id = "get_data_directory",
        python_callable=get_data_dir
    )

    task_drop_created_at_column = PythonOperator(
        task_id='prod_data_preprocess',
        python_callable=drop_created_at_column,
        provide_context=True
    )

    task_data_schema_validation = BranchPythonOperator(
        task_id='if_validate_data_schema',
        python_callable=schema_validation
    )

    task_data_statistics_validation = BranchPythonOperator(
        task_id='if_validate_data_statistics',
        python_callable=stats_validation
    )

    task_prepare_email_schema_validation_failed = PythonOperator(
        task_id='prepare_email_schema_content',
        python_callable=prepare_email_content_schema_prod,
        provide_context=True,
    )

    task_prepare_email_statistics_validation_failed = PythonOperator(
        task_id='prepare_email_stats_content',
        python_callable=prepare_email_content_statistics_prod,
        provide_context=True,
    )

    task_send_email_schema_validation_failed = EmailOperator(
        task_id='email_schema_validation_failed',
        to='rishabkhuba3108@gmail.com',
        subject='Airflow Alert - Batch Retrain Pipeline',
        html_content="{{ task_instance.xcom_pull(task_ids='prepare_email_schema_content') }}"
    )

    task_send_email_statistics_validation_failed = EmailOperator(
        task_id='email_statistics_validation_failed',
        to='rishabkhuba3108@gmail.com',
        subject='Airflow Alert - Batch Retrain Pipeline',
        html_content="{{ task_instance.xcom_pull(task_ids='prepare_email_stats_content') }}"
    )

    # Dummy operator to signify the end of parallel tasks
    task_end = DummyOperator(
        task_id="end_monitor_task",
        trigger_rule = "none_failed"
    )
    

    task_get_data_directory >> task_drop_created_at_column >> task_data_schema_validation 

    # If schema validation succeeds go to statistics validation 
    task_data_schema_validation >> task_data_statistics_validation

    # If statistics validation succeeds end task
    task_data_statistics_validation >> task_end

    # If statistics validation fails airflow creates a task of creating an email for stats validation fail and sends email to admin for data drift
    task_data_statistics_validation >> task_prepare_email_statistics_validation_failed >> task_send_email_statistics_validation_failed

    # If schema validation fails airflow creates a task of creating an email for stats validation fail and sends email to admin for schema changes
    task_data_schema_validation >> task_prepare_email_schema_validation_failed >> task_send_email_schema_validation_failed
