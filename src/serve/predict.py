import os
import re
import pickle
from google.cloud import storage
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from google.cloud import storage
import numpy as np
import pandas as pd
import time
from datetime import datetime
from google.cloud import storage, bigquery, logging
from google.cloud.bigquery import SchemaField

load_dotenv()

app = Flask(__name__)
bucket_name = os.getenv("BUCKET")

bq_client = bigquery.Client()
table_id = os.environ['MODEL_MONITORING_TABLE_ID']

client = logging.Client()
logger = client.logger('Serving_Pipeline')
logger.log_text(f'Iniatiating Serving Pipeline {datetime.now().isoformat()}', severity='INFO')

def create_logging_table_schema():
    """Build the table schema for the logging table
        
        Returns:
            List: List of `SchemaField` objects"""
    return [
        SchemaField("n_input_rows", "INTEGER", mode="REQUIRED"),
        SchemaField("predictions", "STRING", mode="REQUIRED"),
        SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        SchemaField("total_latency", "FLOAT", mode="REQUIRED"),
        SchemaField("average_latency_per_row", "FLOAT", mode="REQUIRED"),
    ]

def create_or_get_logging_table_bq(client, table_id):
    """Create a Logging table in BQ if it doesn't exist
    
    Args:
        client (bigquery.client.Client): A BigQuery Client
        table_id (str): The ID of the table to create        
    Returns:
        None"""
    try:
        client.get_table(table_id)  # Make an API request.
        print(f"Table {table_id} already exists.")
    except Exception as e:
        print(e)
        print(f"Table {table_id} not found. Creating table...")
        logger.log_text(f"Table {table_id} not found. Creating table...", severity='INFO')
        schema = create_logging_table_schema()
        table = bigquery.Table(table_id, schema)
        client.create_table(table)
        logger.log_text(f"Created table {table.project}.{table.dataset_id}.{table.table_id}", severity='INFO')
        print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")

def initialize_client_and_bucket(bucket_name=bucket_name):
    """
    Initialize a storage client and get a bucket object.
    Args:
        bucket_name (str): The name of the bucket.
    Returns:
        tuple: The storage client and bucket object.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    return storage_client, bucket

def load_pickle_from_bucket(bucket, pickle_file_path):
    local_temp_file = "temp.pkl"
    blob = bucket.blob(pickle_file_path)
    blob.download_to_filename(local_temp_file)

    with open(local_temp_file, 'rb') as file:
        item = pickle.load(file)

    os.remove(local_temp_file)
    return item

def load_scaler(bucket, pickle_file_path='artifacts/scaler.pkl'):
    scaler = load_pickle_from_bucket(bucket, pickle_file_path)
    print("Downloaded Scaler")
    logger.log_text("Downloaded Scaler", severity='INFO')
    return scaler

def load_model(bucket, models_prefix='models'):
    blobs = list(bucket.list_blobs(prefix=models_prefix))
    model_folders = {}
    for blob in blobs:
        match = re.search(r'model-run-(\d+)-(\d+)', blob.name)
        if match:
            print("Match Name: ", blob.name)
            timestamp = int(match.group(1) + match.group(2))  # Concatenate the timestamp
            model_folders[timestamp] = blob.name.split('/')[1]  # Extract folder name

    if not model_folders:
        logger.log_text("No model folders found in the specified bucket and prefix.", severity='ERROR')
        raise Exception("No model folders found in the specified bucket and prefix.")

    latest_model_folder = model_folders[max(model_folders.keys())]
    logger.log_text(f"Latest Model: {latest_model_folder}", severity='INFO')
    print("Latest Model: ", latest_model_folder)
    
    model_dir = f'models/{latest_model_folder}/model.pkl'
    print("Model Directory: ", model_dir)
    logger.log_text(f"Model Directory: {model_dir}", severity='INFO')
    model = load_pickle_from_bucket(bucket, model_dir)
    print("Loaded Model")
    logger.log_text(f"Loaded Model", severity='INFO')

    return model

def data_preprocess_pipeline(features):
    df = pd.DataFrame(features, columns=['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
    'HospAdmTime', 'ICULOS', 'Patient_ID'])
    columns_to_drop = ['SBP', 'DBP', 'EtCO2', 'BaseExcess', 'HCO3',
                        'pH', 'PaCO2', 'Alkalinephos', 'Calcium',
                        'Magnesium', 'Phosphate', 'Potassium', 'PTT',
                        'Fibrinogen', 'Unit1', 'Unit2']
    df["Unit"] = df["Unit1"] + df["Unit2"]
    df.drop(columns=columns_to_drop, inplace=True)



    grouped_by_patient = df.groupby('Patient_ID')

    # Impute missing values with forward and backward fill
    df = grouped_by_patient.apply(lambda x: x.bfill().ffill()).reset_index(drop=True)

    # Drop columns with more than 25% null values and 'Patient_ID'
    columns_with_nulls = ['TroponinI', 'Bilirubin_direct', 'AST', 'Bilirubin_total',
                            'Lactate', 'SaO2', 'FiO2', 'Unit', 'Patient_ID']
    df.drop(columns=columns_with_nulls, inplace=True)

    # Apply log transformation to normalize specific columns
    columns_to_normalize = ['MAP', 'BUN', 'Creatinine', 'Glucose', 'WBC', 'Platelets']
    for col in columns_to_normalize:
        if df[col].notna().all():
            print(df[col].values)
            df[col] = np.log1p(df[col])


    # One-hot encode the 'Gender' column
    
    df['gender'] = df['Gender'].replace({1: 'M', 0: 'F'}).apply(lambda x: x if x in ['M', 'F'] else np.nan)
    
    
    encoded_gender_col = pd.get_dummies(df["gender"])
    
    if 'F' not in encoded_gender_col:
        encoded_gender_col['F'] = 0
    if 'M' not in encoded_gender_col:
        encoded_gender_col['M'] = 0

    df = df.join(encoded_gender_col)
    df = df.drop(columns=["Gender", "gender"], axis=1)
    
    df['F'] = df['F'].astype(int)
    df['M'] = df['M'].astype(int)
    # Drop remaining rows with any NaN values
    df.dropna(inplace=True)


    # Split the dataframe back into X and y
    X_preprocessed = df.reset_index(drop=True)

    X_preprocessed = X_preprocessed.drop(columns=X_preprocessed.columns[X_preprocessed.columns.str.contains('^Unnamed', case=False, regex=True)])
    columns_to_scale = ['HR', 'O2Sat', 'Temp', 'MAP', 'Resp', 'BUN', 'Chloride', 'Creatinine', 'Glucose', 'Hct', 'Hgb', 'WBC', 'Platelets']
    X_preprocessed[columns_to_scale] = scaler.transform(X_preprocessed[columns_to_scale])
    X_preprocessed_scaled = scaler.transform(X_preprocessed)
    return X_preprocessed_scaled
        
@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """Health check endpoint that returns the status of the server.
    Returns:
        Response: A Flask response with status 200 and "healthy" as the body.
    """
    return {"status": "The app is healthy"}

@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict():
    """
    Prediction route that normalizes input data, and returns model predictions.
    Returns:
        Response: A Flask response containing JSON-formatted predictions.
    """
    request_json = request.get_json()

    # Prediction Start
    prediction_start_time = time.time()
    current_timestamp = datetime.now().isoformat()

    if not request_json:
        return jsonify({"error": "Invalid input, no JSON payload provided"}), 400
    input_data = request_json.get('data')
    if input_data is None:
        return jsonify({"error": "Invalid input, 'data' field is missing"}), 400

    input_array = np.array(input_data)
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)
    print(input_array.shape)
    logger.log_text(f"Array Shape: {input_array.shape}", severity='INFO')
    if input_array.shape[1]!= 41:
        return jsonify({"error": "Invalid input shape"}), 400
    features = input_array.copy()
    input_array = data_preprocess_pipeline(features)
    predictions = model.predict(input_array)

    # Prediction End
    prediction_end_time = time.time()
    prediction_latency = prediction_end_time - prediction_start_time

    # Insert rows to BQ
    log_latency_data = [{
        "n_input_rows" : len(input_array),
        "predictions": ", ".join([str(pred) for pred in predictions.tolist()]),
        "timestamp": current_timestamp,
        "total_latency": prediction_latency,
        "average_latency_per_row":prediction_latency / len(input_array),
    }]
    errors = bq_client.insert_rows_json(table_id, log_latency_data)
    if errors == []:
        logger.log_text("New predictions inserted into Logging table BQ.", severity='INFO')
    else:
        logger.log_text(f"Encountered errors inserting predictions into Logging table BQ.: {errors}", severity='ERROR')

    return jsonify({"predictions": predictions.tolist()})

    
# Workflow
create_or_get_logging_table_bq(bq_client, table_id)
storage_client, bucket = initialize_client_and_bucket()
scaler = load_scaler(bucket=bucket)
model = load_model(bucket=bucket)

if __name__ == '__main__':
    print("Started predict.py ")
    app.run(host='0.0.0.0', port=os.environ["AIP_HTTP_PORT"])
