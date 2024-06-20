from typing import List
from airflow import DAG
import os
import sys
from airflow.utils.task_group import TaskGroup
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom imports
import dags.utils.config as config
from dags.utils.helper import clean_pickle_files, revert_merge_on_task_fail
from dags.utils.data_preprocessing import data_preprocess_pipeline 
from dags.utils.data_split_utils import train_test_split
from dags.utils.data_scale_utils import scale_train_data, scale_test_data

def data_processing_task_group(dag, data_path, train_type):
    with TaskGroup(group_id='data_processing_and_saving', dag=dag) as paths:
        task_train_test_split = PythonOperator(
            task_id='train_test_split',
            python_callable=train_test_split,
            op_kwargs={'data_path': data_path},
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
            op_kwargs={'data_pkl': 'X_train_processed.pkl', 'scaler_pkl': 'scaler.pkl','output_pkl':'X_train_processed_scaled.pkl'}
        )

        task_scale_test_data = PythonOperator(
            task_id='scale_test_data',
            python_callable=scale_test_data,
            op_kwargs={'data_pkl': 'X_test_processed.pkl', 'scaler_pkl': 'scaler.pkl','output_pkl':'X_test_processed_scaled.pkl'}
        )

        task_push_scaler = LocalFilesystemToGCSOperator(
            task_id="push_scaler_to_gcs",
            gcp_conn_id=config.GCP_CONN_ID,
            src="scaler.pkl",
            dst="artifacts/scaler.pkl",
            bucket=config.bucket
        )

        task_push_X_train_data = LocalFilesystemToGCSOperator(
            task_id="push_X_train_data_to_gcs",
            gcp_conn_id=config.GCP_CONN_ID,
            src="X_train_processed_scaled.pkl",
            dst="data/processed_data/X_train.pkl",
            bucket=config.bucket
        )

        task_push_X_test_data = LocalFilesystemToGCSOperator(
            task_id="push_X_test_data_to_gcs",
            gcp_conn_id=config.GCP_CONN_ID,
            src="X_test_processed_scaled.pkl",
            dst="data/processed_data/X_test.pkl",
            bucket=config.bucket
        )

        task_push_y_train_data = LocalFilesystemToGCSOperator(
            task_id="push_y_train_data_to_gcs",
            gcp_conn_id=config.GCP_CONN_ID,
            src="y_train.pkl",
            dst="data/processed_data/y_train.pkl",
            bucket=config.bucket
        )

        task_push_y_test_data = LocalFilesystemToGCSOperator(
            task_id="push_y_test_data_to_gcs",
            gcp_conn_id=config.GCP_CONN_ID,
            src="y_test.pkl",
            dst="data/processed_data/y_test.pkl",
            bucket=config.bucket
        )

        task_cleanup_files = PythonOperator(
            task_id="clean_pickle_files",
            python_callable=clean_pickle_files,
            op_kwargs={"directory": os.getcwd()}
        )

        task_train_test_split >> [task_X_train_data_preprocessing, task_X_test_data_preprocessing] >> task_scale_train_data >> task_scale_test_data >> [task_push_scaler, task_push_X_train_data, task_push_X_test_data, task_push_y_train_data, task_push_y_test_data] >> task_cleanup_files

        if train_type == "batch_train":
            task_track_error = PythonOperator(
                task_id='revert_merge_if_error',
                python_callable = revert_merge_on_task_fail,
                trigger_rule="one_failed"
            )
            [task_train_test_split, task_X_train_data_preprocessing, task_X_test_data_preprocessing, task_scale_train_data, task_scale_test_data, task_push_scaler, task_push_X_train_data, task_push_X_test_data, task_push_y_train_data, task_push_y_test_data,task_cleanup_files] >> task_track_error

    return paths


