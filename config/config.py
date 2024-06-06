#  this is the configuration file, all the settings like model hyper parameters, storage links are stored here
import numpy as np
from sklearn.pipeline import make_pipeline

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

