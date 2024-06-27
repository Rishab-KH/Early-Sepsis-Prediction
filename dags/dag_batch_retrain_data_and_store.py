from airflow import DAG
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import re
import json
import sys
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.operators.email import EmailOperator
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from google.cloud import storage
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom imports
import utils.config as config
from utils.helper import prepare_email_content, save_data_to_pickle, load_data_from_pickle
from utils.schema_stats_utils import schema_and_stats_validation
from utils.data_preprocessing import data_preprocess_pipeline 
from utils.data_scale_utils import scale_test_data
from include.factory_data_processing import data_processing_task_group

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

def save_data_pickle(ti):
    try:
        df = pd.read_csv(ti.xcom_pull('get_data_location'), sep=",")

        X = df.drop('SepsisLabel', axis=1)
        y = df['SepsisLabel']

        save_data_to_pickle(X, 'X.pkl')
        save_data_to_pickle(y, 'y.pkl')

    except Exception as ex:
        print(f"Saving data failed with an exception as {ex}")

def download_scaler():
    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(config.bucket)
    blob = bucket.blob("artifacts/scaler.pkl")
    blob.download_to_filename("scaler.pkl")
    print("Downloaded Scaler")

def download_latest_model():
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(config.bucket)
    blobs = list(bucket.list_blobs(prefix="models"))
    model_folders = {}
    for blob in blobs:
        match = re.search(r'model-run-(\d+)-(\d+)', blob.name)
        if match:
            print("Match Name: ", blob.name)
            timestamp = int(match.group(1) + match.group(2))  # Concatenate the timestamp
            model_folders[timestamp] = blob.name.split('/')[1]  # Extract folder name

    if not model_folders:
        raise Exception("No model folders found in the specified bucket and prefix.")

    latest_model_folder = model_folders[max(model_folders.keys())]
    print("Latest Model: ", latest_model_folder)
    
    model_dir = f'models/{latest_model_folder}/model.pkl'
    print("Model Directory: ", model_dir)
    blob = bucket.blob(model_dir)
    blob.download_to_filename("model.pkl")
    print("Downloaded Model")

    metrics_dir = f'models/{latest_model_folder}/metrics.json'
    print("Metrics Directory: ", metrics_dir)
    blob = bucket.blob(metrics_dir)
    str_json = blob.download_as_text()
    metrics = json.loads(str_json)
    return metrics

def execute_model_and_get_results():
    X = load_data_from_pickle("X_processed_scaled.pkl")
    y_val = load_data_from_pickle("y.pkl")
    model = load_data_from_pickle("model.pkl")
    print(y_val.value_counts())
    y_pred = model.predict(X)

    y_val = y_val.to_numpy() if isinstance(y_val, pd.Series) else y_val
    y_pred = y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_val, y_pred)
    class_report = classification_report(y_val, y_pred)

    metrics = {'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,}
    print(metrics)

    return metrics # pull later using xcom

def track_model_drift(ti):
    batch_matrics = ti.xcom_pull('execute_model_and_get_results')
    current_matrics = ti.xcom_pull('download_latest_model')
    print(batch_matrics)
    print(current_matrics)
    
    perc_change_in_recall = 100*(current_matrics['recall'] - batch_matrics['recall']) / current_matrics['recall']

    if perc_change_in_recall > 3:
        print(f"Warning: Change in recall: {perc_change_in_recall}% is more than 5%, model might be drifting")
    if perc_change_in_recall > 5:
        print(f"Critical: Heavy change in recall: {perc_change_in_recall}% is more than 10%, retraining model with new batch data")
        return 'trigger_model_retrain'
    return 'set_batch_number'



def helper_merge_df_and_push_to_gcs(bucket, source_data_dir, new_batch_data_dir):
    df = pd.read_csv(source_data_dir, sep=",")
    batch_df = pd.read_csv(new_batch_data_dir, sep=",")

    combined_df = pd.concat([df, batch_df], ignore_index=True)

    client = storage.Client().create_anonymous_client()
    bucket = client.get_bucket(config.bucket)
    bucket.blob('data/modified_data/processed_batch/combined_data.csv').upload_from_string(combined_df.to_csv(index=False), 'text/csv')

def merge_batch_and_existing_data(ti):
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(config.bucket)
    current_batch = ti.xcom_pull('get_batch_number')
    if current_batch == "batch-1":
        print("Data was never batched before, combining initial data and batch-1")
        source_data_dir = config.DATA_DIR
    else:
        print("Data was previously batched, using existing combined df")
        source_data_dir = f"{config.gsutil_URL}/data/modified_data/processed_batch/combined_data.csv"

        print("Saving a backup file for revert if future tasks error out")
        source_blob = bucket.get_blob('data/modified_data/processed_batch/combined_data.csv')
        destination_blob_name = 'data/modified_data/processed_batch/backup/combined_data.csv'
        blob_copy = bucket.copy_blob(source_blob, bucket, new_name=destination_blob_name)
        print(f"Backup created from folder data/modified_data/processed_batch/combined_data.csv to {destination_blob_name} within bucket {bucket}")

    new_batch_data_dir = ti.xcom_pull('get_data_location')
    helper_merge_df_and_push_to_gcs(bucket, source_data_dir, new_batch_data_dir)

with DAG(
    dag_id = "batch_train_model_data_and_store",
    description = "This DAG is responsible for training in batches",
    start_date =datetime(2024,5,15,2),
    schedule_interval = None,
    default_args=default_args,
    catchup = False,
    template_searchpath=["/opt/airflow/dags/utils","/home/airflow/gcs/dags/utils"]
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

    task_save_data_pickle = PythonOperator(
        task_id='download_data',
        python_callable=save_data_pickle,
        trigger_rule='none_failed'
    )

    task_download_scaler = PythonOperator(
        task_id='download_scaler',
        python_callable=download_scaler,
        trigger_rule='none_failed'
    )

    task_download_latest_model = PythonOperator(
        task_id='download_latest_model',
        python_callable=download_latest_model,
        trigger_rule='none_failed'
    )

    task_batch_data_preprocessing = PythonOperator(
        task_id='preprocess_batch_data',
        python_callable=data_preprocess_pipeline,
        op_kwargs={'data_input': 'X.pkl', 'target_input': 'y.pkl', 'data_output':'X_processed.pkl'}
    )

    task_scale_data = PythonOperator(
        task_id='scale_data',
        python_callable=scale_test_data,
        op_kwargs={'data_pkl': 'X_processed.pkl', 'scaler_pkl': 'scaler.pkl','output_pkl':'X_processed_scaled.pkl'}
    )

    task_execute_model_and_get_results = PythonOperator(
        task_id='execute_model_and_get_results',
        python_callable=execute_model_and_get_results
    )

    task_merge_batch_and_existing_data = PythonOperator(
        task_id='merge_batch_and_existing_data',
        python_callable=merge_batch_and_existing_data
    )

    task_track_model_drift = BranchPythonOperator(
        task_id = "track_model_drift",
        python_callable = track_model_drift,
        trigger_rule = 'none_failed'
    )

    task_set_batch_number_to_process = PythonOperator(
        task_id = "set_batch_number",
        python_callable = set_next_batch_folder,
        trigger_rule = 'none_failed'
    )

    task_trigger_modelling_dag = TriggerDagRunOperator(
        task_id="trigger_model_retrain",
        trigger_dag_id="model_data_and_store",
        trigger_rule = 'none_failed'
    )

    task_get_batch_number_to_process >> task_batch_gcs_psv_to_gcs_csv >> task_get_data_directory >> task_data_schema_and_statastics_validation
    task_data_schema_and_statastics_validation >> task_prepare_email_validation_failed >> task_send_email_validation_failed
    task_data_schema_and_statastics_validation >> [task_download_scaler, task_save_data_pickle, task_download_latest_model] >> task_batch_data_preprocessing >> task_scale_data >> task_execute_model_and_get_results >> task_merge_batch_and_existing_data >> data_processing_task_group(dag, f"{config.gsutil_URL}/data/modified_data/processed_batch/combined_data.csv", train_type='batch_train') >> task_track_model_drift >> task_set_batch_number_to_process
    task_track_model_drift >> task_trigger_modelling_dag >> task_set_batch_number_to_process