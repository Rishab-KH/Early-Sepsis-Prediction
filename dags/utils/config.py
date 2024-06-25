import os
from datetime import datetime

#  this is the configuration file, all the settings like model hyper parameters, storage links are stored here
# GCP bucket URL where the data is saved
gsutil_URL= 'gs://sepsis-prediction-mlops'

# bucket name
bucket = 'sepsis-prediction-mlops'
output_bucket = 'sepsis-prediction-outputs'

# data directory
DATA_DIR = 'gs://sepsis-prediction-mlops/data/modified_data/finalDataset-000000000000.csv'

#predict directory
PREDICT_DIR = "gs://sepsis-prediction-mlops/data/modified_data/prod_data/ProdDataset.csv"

# name of the preprocessed data pickle file
PREPROCESSED_DATA='preprocessed_data.pickle'

# number of samples to be used while preprocessing the data
NUM_SAMPLES=30000

# test sample size
TEST_SAMPLE_SIZE=2000

# filename in gcp for schema and stats
STATS_SCHEMA_FILE = 'schema_and_stats.json'

# stored artifacts i.e, schema and stats json file loc in gcp
STATS_SCHEMA_FILE_GCS = 'gs://sepsis-prediction-mlops/artifacts/schema_and_stats.json'

# project root folder
PROJECT_ROOT = os.path.abspath(os.environ["AIRFLOW_HOME"])

# project name
GCP_PROJECT_NAME = "leafy-sunrise-425218-h4"

# connection id
GCP_CONN_ID = "google_cloud_default"

# mlflow_uri
TRACKING_URI = "https://mlflow-cloud-run-3wcd2ryf5q-uc.a.run.app"

# artifact storage name
artifact_base_path = f'model-run-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

