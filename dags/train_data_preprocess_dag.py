from airflow import DAG
from datetime import datetime, timedelta
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.operators.s3 import S3ListOperator
import os
from zipfile import ZipFile 
import pandas as pd

default_args = {
    "owner": 'airflow',
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

def download_from_s3() -> None:
    hook = S3Hook('my-aws-conn')
    file_name = hook.download_file(
        key='data/initial_training/Archive.zip',
        bucket_name='sepsis-training-bucket'
    )
    return file_name

def rename_file_and_extract(ti, new_name: str):
    list_of_args = ti.xcom_pull(task_ids=['download_from_s3'])
    downloaded_file_name = list_of_args[0]
    downloaded_file_path = '/'.join(downloaded_file_name.split('/')[:-1])
    new_name_for_file = f'{downloaded_file_path}/{new_name}'
    os.rename(src=downloaded_file_name, dst=new_name_for_file)
    new_file_path = f'{downloaded_file_path}/train_preprocess/'

    with ZipFile(new_name_for_file, 'r') as zObject: 
        zObject.extractall(path=new_file_path) 

    return new_file_path
    
def convert_psv_to_df(ti):
    list_of_args = ti.xcom_pull(task_ids=['rename_and_unzip_file'])
    filepath = list_of_args[0]
    count = 0
    rows = 0
    datatable = pd.DataFrame()
    for filename in os.listdir(filepath):
        if filename.endswith(".psv"): 
            with open(filepath + filename) as openfile:
                patient = filename.split("p")[1]
                patient = patient.split(".")[0]

                file = pd.read_csv(openfile,sep = "|")
                file['Patient_ID'] = patient
                
                file = file.reset_index()
                file = file.rename(columns={"index": "Hour"})
                
                datatable = pd.concat([datatable, file])
                
                rows += file.size
                count += 1

        if count % 10000 == 0:
            print("Progress || Files: {} || Number of items: {}".format(count,rows))
    print("Done ||| Files: {} || Number of items: {}".format(count,rows))
    datatable.to_csv("train_dataset.csv", index=None)
    return "train_dataset.csv"

def read_df(ti):
    list_of_args = ti.xcom_pull(task_ids=['psv_to_df'])
    filepath = list_of_args[0]

    df = pd.read_csv(filepath)

    ## DATA PREPROCESSING BEGINS HERE
    print(df)

with DAG(
    dag_id = "train_data_preprocess",
    description = "This DAG is responsible for cleaning and preprocessing the raw data into usable format",
    start_date =datetime(2024,5,15,2),
    schedule_interval = None,
    default_args=default_args,
    catchup = False
) as dag:
    task_download_from_s3 = PythonOperator(
        task_id='download_from_s3',
        python_callable=download_from_s3
    )

    task_unzip_file = PythonOperator(
        task_id='rename_and_unzip_file',
        python_callable=rename_file_and_extract,
        op_kwargs={'new_name': 'train_data_preprocess.zip'}
    )

    task_convert_psv_to_df = PythonOperator(
        task_id='psv_to_df',
        python_callable=convert_psv_to_df
    )

    task_dummy_read_df = PythonOperator(
        task_id='read_df_dummy',
        python_callable=read_df
    )
    
    task_download_from_s3 >> task_unzip_file >> task_convert_psv_to_df >> task_dummy_read_df