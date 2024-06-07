import os

#  this is the configuration file, all the settings like model hyper parameters, storage links are stored here
# GCP bucket URL where the data is saved
gsutil_URL= 'gs://sepsis-prediction-mlops'

# Data Directory
DATA_DIR = "gs://sepsis-prediction-mlops/data/modified_data/finalDataset-000000000000.csv"

# name of the preprocessed data pickle file
PREPROCESSED_DATA='preprocessed_data.pickle'

# number of samples to be used while preprocessing the data
NUM_SAMPLES=30000

# test sample size
TEST_SAMPLE_SIZE=2000

STATS_SCHEMA_FILE = 'schema_and_stats.json'
STATS_SCHEMA_FILE_GCS = 'gs://sepsis-prediction-mlops/artifacts/schema_and_stats.json'

PROJECT_ROOT = os.path.abspath(os.environ["AIRFLOW_HOME"])

GCP_PROJECT_NAME = "leafy-sunrise-425218-h4"