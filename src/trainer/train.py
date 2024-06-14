# Import libraries
import os, logging, argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error
import pickle
import gcsfs
from datetime import datetime

from google.cloud import storage

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Setup logging and parser
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

# Input Arguments
parser.add_argument(
    '--gcs_bucket_path',
    help = 'Bucket where files are stored',
    type = str
)

parser.add_argument(
    '--model_dir',
    help = 'Directory to output model artifacts',
    type = str,
    default = os.environ['AIP_MODEL_DIR'] if 'AIP_MODEL_DIR' in os.environ else ""
)

# Parse argumentsss
args = parser.parse_args()
arguments = args.__dict__

def load_data_from_gcs(file_paths):
    fs = gcsfs.GCSFileSystem()
    
    data = {}
    for key, file_path in file_paths.items():
        with fs.open(file_path, 'rb') as f:
            data[key] = pickle.load(f)
    
    return data

def evaluate_model(y_true,y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    precision = precision_score(y_true, y_pred)
    print("Precision:", precision)
    recall = recall_score(y_true, y_pred)
    print("Recall:", recall)
    f1 = f1_score(y_true, y_pred)
    print("F1 Score:", f1)
    auc = roc_auc_score(y_true, y_pred)
    print("AUC-ROC:", auc)
    mae = mean_absolute_error(y_true, y_pred)
    print("Mean Absolute Error:", mae)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("Root Mean Squared Error:", rmse)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

file_paths = {
    'X_train': f"gs://{arguments['gcs_bucket_path']}/X_train.pkl",
    'X_test': f"gs://{arguments['gcs_bucket_path']}/X_test.pkl",
    'y_train': f"gs://{arguments['gcs_bucket_path']}/y_train.pkl",
    'y_test': f"gs://{arguments['gcs_bucket_path']}/y_test.pkl"
}

data = load_data_from_gcs(file_paths)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

logging.info("reading gs data: {}".format(f"{arguments['gcs_bucket_path']}"))

# Combine (THIS IS TEMPORARY)
X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
train_df = pd.concat([X, y], axis=1)


# Rishab Fix this
majority_class = train_df[train_df['SepsisLabel'] == 0]
minority_class = train_df[train_df['SepsisLabel'] == 1]

majority_class_subset = majority_class.sample(n=2*len(minority_class))
train_df = pd.concat([majority_class_subset, minority_class])

X_train = train_df.drop('SepsisLabel', axis=1)
y_train = train_df['SepsisLabel']

X = train_df.drop('SepsisLabel', axis=1)
y = train_df['SepsisLabel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=300, random_state=0)
model.fit(X_train, y_train)
rcf_predictions = model.predict(X_test)

evaluate_model(y_test,rcf_predictions)

# Define model name
artifact_filename = f'model-{datetime.now().strftime("%Y%m%d-%H%M%S")}.pkl'

# Save model artifact to local filesystem (doesn't persist)
local_path = artifact_filename
with open(local_path, 'wb') as file:
    pickle.dump(model, file)

# Upload model artifact to Cloud Storage
model_directory = arguments['model_dir']
if model_directory == "":
    print("Training is run locally - skipping model saving to GCS.")
else:
    storage_path = os.path.join(model_directory, artifact_filename)
    print(storage_path)
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_filename(local_path)
    logging.info("model exported to : {}".format(storage_path))