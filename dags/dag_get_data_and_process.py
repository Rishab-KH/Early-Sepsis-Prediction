from airflow import DAG
from datetime import datetime, timedelta
import pandas as pd
import pickle
import os
import sys
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import logging

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom imports
import dags.utils.config as config
from dags.utils.data_preprocessing import util_data_preprocessing 
from dags.utils.log_config import setup_logging
from dags.utils.Data_Validation import generate_and_save_schema_and_stats,validate_data

DATA_DIR = config.DATA_DIR
STATS_SCHEMA_FILE = config.STATS_SCHEMA_FILE

logger = setup_logging(config.PROJECT_ROOT, "dag_get_data_and_preprocess.py")

# logger = logging.getLogger('DAG_Get_Data_and_Preprocess.py')



default_args = {
    "owner": 'airflow',
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

def save_data_to_pickle(obj, obj_name):
    with open(obj_name, 'wb') as file:
        pickle.dump(obj, file)

def load_data_from_pickle(obj_name):
    with open(obj_name, 'rb') as file:
        obj = pickle.load(file)
    return obj


def schema_stats_gen():
    try:
        df=pd.read_csv(DATA_DIR, sep=",")
        logger.info(f"Data loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"File not found: {DATA_DIR}. Error: {e}")
        raise ValueError("Failed to Load Data for Schema and Statstics Validation. Stopping DAG execution.")
    #STATS_SCHEMA_FILE = 'schema_and_stats.json'
    generation_result=generate_and_save_schema_and_stats(df, STATS_SCHEMA_FILE)
    if not generation_result:
        raise ValueError("Schema and Statstics Generation failed. Stopping DAG execution.")

def schema_and_stats_validation():
    try:
        df=pd.read_csv(DATA_DIR, sep=",")
        logger.info(f"Data loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"File not found: {DATA_DIR}. Error: {e}")
        raise ValueError("Failed to Load Data for Schema and Statstics Validation. Stopping DAG execution.")
    validation_result=validate_data(df)
    if not validation_result:
        raise ValueError("Schema and Statstics Validation failed. Stopping DAG execution.")


def train_test_split():
    df = pd.read_csv(DATA_DIR, sep=",")
    train_inds, test_inds = next(GroupShuffleSplit(test_size=0.25, n_splits=2,).split(df, groups=df['Patient_ID']))
    df_train = df.iloc[train_inds] 
    df_test = df.iloc[test_inds]

    X_train = df_train.drop('SepsisLabel', axis=1)
    X_test = df_test.drop('SepsisLabel', axis=1)
    y_train = df_train['SepsisLabel']
    y_test = df_test['SepsisLabel']

    save_data_to_pickle(X_train, 'X_train.pkl')
    save_data_to_pickle(X_test, 'X_test.pkl')
    save_data_to_pickle(y_train, 'y_train.pkl')
    save_data_to_pickle(y_test, 'y_test.pkl')

def data_preprocess(data_input, data_output):
    X_df = load_data_from_pickle(data_input)
    X_preprocessed_df = util_data_preprocessing(X_df)
    save_data_to_pickle(X_preprocessed_df, data_output)

def scale_train_data():
    X_train_processed_df = load_data_from_pickle('X_train_processed.pkl')
    scaler = StandardScaler()
    columns_to_scale = ['HR', 'O2Sat', 'Temp', 'MAP', 'Resp', 'BUN', 'Chloride', 'Creatinine', 'Glucose', 'Hct', 'Hgb', 'WBC', 'Platelets']
    X_train_processed_df[columns_to_scale] = scaler.fit_transform(X_train_processed_df[columns_to_scale])
    
    save_data_to_pickle(X_train_processed_df, 'X_train_processed_scaled.pkl')
    save_data_to_pickle(scaler, 'scaler.pkl')

def scale_test_data():
    X_test_processed_df = load_data_from_pickle('X_test_processed.pkl')
    scaler = load_data_from_pickle('scaler.pkl')

    columns_to_scale = ['HR', 'O2Sat', 'Temp', 'MAP', 'Resp', 'BUN', 'Chloride', 'Creatinine', 'Glucose', 'Hct', 'Hgb', 'WBC', 'Platelets']
    X_test_processed_df[columns_to_scale] = scaler.transform(X_test_processed_df[columns_to_scale])

    save_data_to_pickle(X_test_processed_df, 'X_test_processed_scaled.pkl')

def clean_pickle_files():
    directory = os.getcwd()
    files = os.listdir(directory)
    for file in files:
        if file.endswith('.pkl'):
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            print(f"Deleted {file_path}")

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
    gcp_conn_id="my-gcp-conn",
    configuration={
        "query": {
            "query": "{% include '/merge_data_from_psv_to_csv.sql' %}",
            "useLegacySql": False
        }
    }
    )

    task_Schema_and_Statastics_Generation = PythonOperator(
        task_id='schema_and_statstics_generation',
        python_callable=schema_stats_gen
    )

    task_push_generated_schema_data = LocalFilesystemToGCSOperator(
       task_id="push_generated_schema_data_to_gcs",
       gcp_conn_id="my-gcp-conn",
       src=STATS_SCHEMA_FILE,
       dst=f"artifacts/{STATS_SCHEMA_FILE}",
       bucket="sepsis-prediction-mlops"
    )

    task_Data_Schema_and_Statastics_Validation = PythonOperator(
        task_id='data_schema_and_statastics_validation',
        python_callable=schema_and_stats_validation

    )   

    task_train_test_split = PythonOperator(
        task_id='train_test_split',
        python_callable=train_test_split
    )

    task_X_train_data_preprocessing = PythonOperator(
        task_id='preprocess_X_train',
        python_callable=data_preprocess,
        op_kwargs={'data_input': 'X_train.pkl', 'data_output':'X_train_processed.pkl'}
    )

    task_X_test_data_preprocessing = PythonOperator(
        task_id='preprocess_X_test',
        python_callable=data_preprocess,
        op_kwargs={'data_input': 'X_test.pkl', 'data_output':'X_test_processed.pkl'}
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
       gcp_conn_id="my-gcp-conn",
       src="scaler.pkl",
       dst="artifacts/scaler.pkl",
       bucket="sepsis-prediction-mlops"
    )

    task_push_X_train_data = LocalFilesystemToGCSOperator(
       task_id="push_X_train_data_to_gcs",
       gcp_conn_id="my-gcp-conn",
       src="X_train_processed_scaled.pkl",
       dst="data/processed_data/X_train.pkl",
       bucket="sepsis-prediction-mlops"
    )

    task_push_X_test_data = LocalFilesystemToGCSOperator(
       task_id="push_X_test_data_to_gcs",
       gcp_conn_id="my-gcp-conn",
       src="X_test_processed_scaled.pkl",
       dst="data/processed_data/X_test.pkl",
       bucket="sepsis-prediction-mlops"
    )

    task_push_y_train_data = LocalFilesystemToGCSOperator(
       task_id="push_y_train_data_to_gcs",
       gcp_conn_id="my-gcp-conn",
       src="y_train.pkl",
       dst="data/processed_data/y_train.pkl",
       bucket="sepsis-prediction-mlops"
    )

    task_push_y_test_data = LocalFilesystemToGCSOperator(
       task_id="push_y_test_data_to_gcs",
       gcp_conn_id="my-gcp-conn",
       src="y_test.pkl",
       dst="data/processed_data/y_test.pkl",
       bucket="sepsis-prediction-mlops"
    )

    task_cleanup_files = PythonOperator(
        task_id="clean_pickle_files",
        python_callable=clean_pickle_files,
    )

    task_trigger_modelling_dag = TriggerDagRunOperator(
        task_id="trigger_modelling_dag",
        trigger_dag_id="model_data_and_store",
    )

    task_gcs_psv_to_gcs_csv >> task_Schema_and_Statastics_Generation >> task_push_generated_schema_data >> task_Data_Schema_and_Statastics_Validation >> task_train_test_split >> [task_X_train_data_preprocessing, task_X_test_data_preprocessing] >> task_scale_train_data >> task_scale_test_data >> [task_push_scaler, task_push_X_train_data, task_push_X_test_data, task_push_y_train_data, task_push_y_test_data] >> task_cleanup_files >> task_trigger_modelling_dag