from airflow import DAG
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import os
import pandas as pd
import re
import sys
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.operators.email import EmailOperator
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from google.cloud import storage

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom imports
import dags.utils.config as config
from dags.utils.helper import prepare_email_content, save_data_to_pickle, load_data_from_pickle
from dags.utils.schema_stats_utils import schema_and_stats_validation
from dags.utils.data_preprocessing import data_preprocess_pipeline 
from dags.utils.data_scale_utils import scale_test_data

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

def execute_model_and_get_results():
    X = load_data_from_pickle("X_processed_scaled.pkl")
    y_val = load_data_from_pickle("y.pkl")
    model = load_data_from_pickle("model.pkl")

    y_pred = model.predict(X)
    
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

    # SAVE JSON IF REQUIRED, LATER FOR COMPARISONS
    print(metrics)

def merge_batch_and_existing_data():
    pass

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

    task_track_model_drift = DummyOperator(task_id='track_model_drift')

    task_merge_batch_and_existing_data = PythonOperator(
        task_id='merge_batch_and_existing_data',
        python_callable=merge_batch_and_existing_data
    )

    task_get_batch_number_to_process >> task_batch_gcs_psv_to_gcs_csv >> task_get_data_directory >> task_data_schema_and_statastics_validation
    task_data_schema_and_statastics_validation >> task_prepare_email_validation_failed >> task_send_email_validation_failed
    task_data_schema_and_statastics_validation >> [task_download_scaler, task_save_data_pickle, task_download_latest_model] >> task_batch_data_preprocessing >> task_scale_data >> task_execute_model_and_get_results >> task_track_model_drift >> task_set_batch_number_to_process