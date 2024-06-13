from airflow import DAG
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import os
from airflow.providers.google.cloud.operators.vertex_ai.custom_job import CreateCustomContainerTrainingJobOperator

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom imports
import dags.utils.config as config

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

    create_custom_container_training_job = CreateCustomContainerTrainingJobOperator(
        task_id="train_sepsis_on_vertex_ai",
        staging_bucket=f"sepsis-prediction-outputs/vertexai/train_joboutputs-{str(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))}",
        display_name=f"train_sepsis_on_vertex_ai",
        container_uri="us-central1-docker.pkg.dev/leafy-sunrise-425218-h4/sepsis-mlops-repo/train_pipeline:latest",
        replica_count=1,
        machine_type="n1-standard-4",
        region="us-central1",
        args=['--gcs_bucket_path=sepsis-prediction-mlops/data/processed_data'],
        gcp_conn_id=config.GCP_CONN_ID
    )
    task_dummy_end = DummyOperator(task_id='end_task')

    create_custom_container_training_job >> task_dummy_end