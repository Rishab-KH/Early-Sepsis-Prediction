# Import libraries
import os, logging, argparse
# Import MLFlow for model training
import mlflow 
import pandas as pd
import numpy as np
import pickle
import gcsfs
from datetime import datetime
import time
import json

from google.cloud import storage

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from dotenv import load_dotenv
from xgboost import XGBClassifier

load_dotenv()

# Setup logging and parser
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

# Set tracking URI for MLFlow
mlflow.set_tracking_uri(os.getenv("TRACKING_URI"))

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

file_paths = {
    'X_train': f"gs://{arguments['gcs_bucket_path']}/X_train.pkl",
    'X_test': f"gs://{arguments['gcs_bucket_path']}/X_test.pkl",
    'y_train': f"gs://{arguments['gcs_bucket_path']}/y_train.pkl",
    'y_test': f"gs://{arguments['gcs_bucket_path']}/y_test.pkl"
}

def load_data_from_gcs(file_paths):

    fs = gcsfs.GCSFileSystem()
    data = {}
    for key, file_path in file_paths.items():
        with fs.open(file_path, 'rb') as f:
            data[key] = pickle.load(f)
    return data

def save_pickle_files(path, obj):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def save_json_file(path, obj):
    with open(path, 'w') as file:
        json.dump(obj, file, indent=4)

def pre_process_split_data(data):
    X_train = data['X_train']
    X_val = data['X_test']
    y_train = data['y_train']
    y_val = data['y_test']
    logging.info("reading gs data: {}".format(f"{arguments['gcs_bucket_path']}"))
    X = pd.concat([X_train, X_val])
    y = pd.concat([y_train, y_val])
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    df = pd.concat([X, y], axis=1)
    majority_class = df[df['SepsisLabel'] == 0]
    minority_class = df[df['SepsisLabel'] == 1]
    majority_class_subset = majority_class.sample(n=2*len(minority_class))
    df = pd.concat([majority_class_subset, minority_class])
    X = df.drop(columns='SepsisLabel', axis=1)
    y = df['SepsisLabel']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def train_models(X_train, X_val, y_train, y_val):

    hyperparameter_set = [
        # {'model': RandomForestClassifier(), 'model_name': 'Random_Forest_Classifier', 'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}},
        {'model': XGBClassifier(),  'model_name': 'XGB_Classifier', 'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.2]}},
        {'model': LogisticRegression(max_iter=200),  'model_name': 'Logistic_Regression', 'params': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}}
        ]

    best_candidates = {}

    for _, hyperparams in enumerate(hyperparameter_set):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = f"model_{timestamp}"
        
        with mlflow.start_run():
            grid_search = GridSearchCV(estimator=hyperparams['model'], param_grid=hyperparams['params'], cv=5, scoring='f1')
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            end_time = time.time()
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            y_pred_val = best_model.predict(X_val)
            f1_val = f1_score(y_val, y_pred_val, average='weighted')
            training_time = end_time - start_time
            
            logging.info(f"Best parameters for {hyperparams['model_name']}: {best_params}")
            logging.info(f"Best score for {hyperparams['model_name']}: {grid_search.best_score_}")

            best_candidates[hyperparams['model_name']] = {
                                        'model': best_model,
                                        'params': best_params,
                                        'f1_score': f1_val,
                                        'training_time': training_time
                                        }
            logging.info(f"{hyperparams['model_name']} - Best Params: {best_params}, F1 Score: {f1_val}")
    
    return best_candidates
        
def get_best_model(best_candidates):

    best_model_name = max(best_candidates, key=lambda x: best_candidates[x]['f1_score'])
    best_model = best_candidates[best_model_name]
    
    logging.info(f"Best Model: {best_model_name}")
    logging.info(f"Best Params: {best_model['params']}")
    logging.info(f"F1 Score: {best_model['f1_score']}")
    
    return best_model, best_model_name


def evaluate_best_model(best_model, best_model_name, X_val, y_val):
    best_overall_model = best_model['model']
    # X_train, X_val, y_train, y_val = pre_process_split_data()
    y_pred = best_overall_model.predict(X_val)
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_val, y_pred)
    class_report = classification_report(y_val, y_pred)
    # Log metrics and parameters
    with mlflow.start_run(run_name=best_model_name):
        mlflow.log_param('model_name', best_model_name)
        mlflow.log_params(best_model['params'])

        metrics = {'training_time': best_model['training_time'],
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,}
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_overall_model, best_model_name)
        mlflow.log_text(str(conf_matrix), 'confusion_matrix.txt')
        mlflow.log_text(class_report, 'classification_report.txt')

        logging.info(f"Best overall model: {best_model_name}")
        logging.info(f"Best overall parameters: {best_model['params']}")
        logging.info(f"Best overall training time: {best_model['training_time']}")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"\nConfusion Matrix:\n{conf_matrix}")
        logging.info(f"\nClassification Report:\n{class_report}")

    return best_overall_model, metrics

def save_and_upload_artifacts(best_model, metrics, arguments):

    model_name = "model.pkl"
    metrics_name = "metrics.json"

    local_model_path = model_name
    local_metrics_path = metrics_name

    save_pickle_files(local_model_path, best_model)
    save_json_file(local_metrics_path, metrics)

    # Upload model and results artifact to Cloud Storage
    model_directory = arguments['model_dir']
    model_storage_path = os.path.join(model_directory,model_name)
    results_storage_path = os.path.join(model_directory,metrics_name)

    # # Upload model and results artifact to Cloud Storage
    # model_directory = arguments['model_dir']

    if model_directory == "":
        print("Training is run locally - skipping model saving to GCS.")
    else:
        model_storage_path = os.path.join(model_directory,model_name)
        blob = storage.blob.Blob.from_string(model_storage_path, client=storage.Client())
        blob.upload_from_filename(local_model_path)
        logging.info("model exported to : {}".format(model_storage_path))

        results_storage_path = os.path.join(model_directory,metrics_name)
        blob = storage.blob.Blob.from_string(results_storage_path, client=storage.Client())
        blob.upload_from_filename(local_metrics_path)
        logging.info("metrics exported to : {}".format(results_storage_path))



# Current Worflow
data  = load_data_from_gcs(file_paths)
X_train, X_val, y_train, y_val = pre_process_split_data(data)
best_candidates = train_models(X_train, X_val, y_train, y_val)
best_model, best_model_name = get_best_model(best_candidates)
best_model, metric  = evaluate_best_model(best_model, best_model_name, X_val, y_val)
save_and_upload_artifacts(best_model, metric, arguments)