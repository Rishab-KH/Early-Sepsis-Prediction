import os
import re
import pickle
from google.cloud import storage
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from google.cloud import storage
import numpy as np
import pandas as pd
from google.cloud import storage, logging, bigquery
from google.oauth2 import service_account

load_dotenv()

app = Flask(__name__)
bucket_name = os.getenv("BUCKET")

# service_account_file = 'GCP_SA.json'
# credentials = service_account.Credentials.from_service_account_file(service_account_file)
bq_client = bigquery.Client()
table_id = os.environ['MODEL_MONITORING_TABLE_ID']

def create_or_get_logging_table_bq(client, table_id, schema):
    """Create a Logging table in BQ if it doesn't exist
    
    Args:
        client (bigquery.client.Client): A BigQuery Client
        table_id (str): The ID of the table to create
        schema (List): List of `SchemaField` objects
        
    Returns:
        None"""
    try:
        client.get_table(table_id)  # Make an API request.
        print(f"Table {table_id} already exists.")
    except Exception as e:
        print(e)
        print(f"Table {table_id} not found. Creating table...")
        # table = bigquery.Table(table_id, schema=schema)
        # client.create_table(table)  # Make an API request.
        # print("Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id))

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
        raise Exception("No model folders found in the specified bucket and prefix.")

    latest_model_folder = model_folders[max(model_folders.keys())]
    print("Latest Model: ", latest_model_folder)
    
    model_dir = f'models/{latest_model_folder}/model.pkl'
    print("Model Directory: ", model_dir)
    model = load_pickle_from_bucket(bucket, model_dir)
    print("Loaded Model")

    return model

def data_preprocess_pipeline(features):
    if True:
        print(features.shape)
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

        return X_preprocessed
        
    # except KeyError as ke:
    #     print("KeyError during preprocessing: %s", ke)
    # except ValueError as ve:
    #     print("ValueError during preprocessing: %s", ve)
    # except Exception as ex:
    #     print("An unexpected error occurred during preprocessing: %s", ex)
    #     raise


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
    
    ## PROCESS HERE ##

    if not request_json:
        return jsonify({"error": "Invalid input, no JSON payload provided"}), 400
    input_data = request_json.get('data')
    if input_data is None:
        return jsonify({"error": "Invalid input, 'data' field is missing"}), 400

    input_array = np.array(input_data)
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)
    print(input_array.shape)
    if input_array.shape[1]!= 41:
        return jsonify({"error": "Invalid input shape"}), 400
    features = input_array.copy()
    input_array = data_preprocess_pipeline(features)


    prediction = model.predict(input_array)
    return jsonify({"predictions": prediction.tolist()})

    
# Workflow
create_or_get_logging_table_bq(bq_client, table_id, schema=None)
storage_client, bucket = initialize_client_and_bucket()
scaler = load_scaler(bucket=bucket)
model = load_model(bucket=bucket)



if __name__ == '__main__':
    print("Started predict.py ")
    app.run(host='0.0.0.0', port=os.environ["AIP_HTTP_PORT"])