from airflow import DAG
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta


default_args = {
    "owner": 'airflow',
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

with DAG(
    dag_id = "model_data_and_store",
    description = "This DAG is responsible for modelling",
    start_date =datetime(2024,5,15,2),
    schedule_interval = None,
    default_args=default_args,
    catchup = False,
    template_searchpath=["/opt/airflow/dags/utils"]
) as dag:

    task_dummy_start = DummyOperator(task_id='start_task')
    task_dummy_end = DummyOperator(task_id='end_task')

    task_dummy_start >> task_dummy_end