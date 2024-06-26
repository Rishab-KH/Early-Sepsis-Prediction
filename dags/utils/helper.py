# Import libraries
import pickle
import os
import sys
from google.cloud import storage

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom import
import utils.config as config
from utils.log_config import setup_logging

logger = setup_logging(config.PROJECT_ROOT, "helper.py")

def revert_merge_on_task_fail():
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(config.bucket)

    print("Reverting to backup file since a task errored out")
    source_blob = bucket.get_blob('data/modified_data/processed_batch/backup/combined_data.csv')
    destination_blob_name = 'data/modified_data/processed_batch/combined_data.csv'
    blob_copy = bucket.copy_blob(source_blob, bucket, new_name=destination_blob_name)
    print(f"Reverted to backup file from folder data/modified_data/processed_batch/combined_data.csv to {destination_blob_name} within bucket {bucket}")

def load_data_from_pickle(obj_name):
    '''
    load_data_from_pickle function loads and deserializes an object from a pickle file.

    Args:
        obj_name (str): The path to the pickle file containing the serialized object. 

    Returns:
        obj (pd.DataFrame): The deserialized object that was stored in the pickle file. 

    '''
    with open(obj_name, 'rb') as file:
        obj = pickle.load(file)
    return obj

def save_data_to_pickle(obj, obj_name):
    '''
    save_data_to_pickle function saves and serializes an object from a pickle file.

    Args:
        obj (pd.DaraFrame): The pandas dataframe that needs to be serialized
        obj_name (str): The path to the pickle file containing the serialized object. 

    Returns:
        dumps serialized dataframe as pickle file to the obj_name i.e, specified path 

    '''
    with open(obj_name, 'wb') as file:
        pickle.dump(obj, file)

def clean_pickle_files(directory):
    """
    Deletes all .pkl files in the specified directory.

    Args:
        directory (str): The path to the directory where .pkl files need to be deleted.

    Returns:
        None
    """
    files = os.listdir(directory)
    for file in files:
        if file.endswith('.pkl'):
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            logger.info(f"Deleted {file_path}")

def prepare_email_content(**context):
    ti = context['ti']
    validation_message = ti.xcom_pull(task_ids='if_validate_data_schema_and_stats', key='validation_message')
    
    dag_run = context['dag_run']
    dag_id = dag_run.dag_id
    execution_date = dag_run.execution_date.isoformat()
    task_id = ti.task_id
    owner = ti.task.dag.owner
    
    # Constructing the HTML content for the email.
    html_content = f"""
    <h3>Validation of Schema/Stats Failed</h3>
    <p>Find the error below:</p>
    <p>{validation_message}</p>
    <br>
    <strong>DAG Details:</strong>
    <ul>
        <li>DAG ID: {dag_id}</li>
        <li>Task ID: {task_id}</li>
        <li>Execution Date: {str(execution_date)}</li>
        <li>Owner: {owner}</li>
    </ul>
    <p>This is an automated message from Airflow. Please do not reply directly to this email.</p>
    """
    return html_content


def prepare_email_content_schema_prod(**context):
    ti = context['ti']
    validation_message = ti.xcom_pull(task_ids='if_validate_data_schema', key='validation_schema_message')
    
    dag_run = context['dag_run']
    dag_id = dag_run.dag_id
    execution_date = dag_run.execution_date.isoformat()
    task_id = ti.task_id
    owner = ti.task.dag.owner
    
    # Constructing the HTML content for the email.
    html_content = f"""
    <h3>Schema of serving data needs to be changed</h3>
    <p>Find the error below:</p>
    <p>{validation_message}</p>
    <p>Schema of the serving data needs to be changed by the data team according to the training</p>
    <br>
    <strong>DAG Details:</strong>
    <ul>
        <li>DAG ID: {dag_id}</li>
        <li>Task ID: {task_id}</li>
        <li>Execution Date: {str(execution_date)}</li>
        <li>Owner: {owner}</li>
    </ul>
    <p>This is an automated message from Airflow. Please do not reply directly to this email.</p>
    """
    return html_content

def prepare_email_content_statistics_prod(**context):
    ti = context['ti']
    validation_message = ti.xcom_pull(task_ids='if_validate_data_statistics', key='validation_email_message')
    
    dag_run = context['dag_run']
    dag_id = dag_run.dag_id
    execution_date = dag_run.execution_date.isoformat()
    task_id = ti.task_id
    owner = ti.task.dag.owner
    
    # Constructing the HTML content for the email.
    html_content = f"""
    <h3>Data drift detected on serving data</h3>
    <p>Find the error below:</p>
    <p>{validation_message}</p>
    <p>Data Drift observed between serving and training data. User needs to send more data similar to production data for re-training</p>
    <br>
    <strong>DAG Details:</strong>
    <ul>
        <li>DAG ID: {dag_id}</li>
        <li>Task ID: {task_id}</li>
        <li>Execution Date: {str(execution_date)}</li>
        <li>Owner: {owner}</li>
    </ul>
    <p>This is an automated message from Airflow. Please do not reply directly to this email.</p>
    """
    return html_content