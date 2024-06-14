import os
import re
import pickle
from google.cloud import storage
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from google.cloud import storage


load_dotenv()

app = Flask(__name__)
bucket_name = os.getenv("BUCKET")

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

@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """Health check endpoint that returns the status of the server.
    Returns:
        Response: A Flask response with status 200 and "healthy" as the body.
    """
    return {"status": "healthy"}

@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict():
    """
    Prediction route that normalizes input data, and returns model predictions.
    Returns:
        Response: A Flask response containing JSON-formatted predictions.
    """
    request_json = request.get_json()
    
    ## PROCESS HERE ##

    prediction = f"you got sepsis bruh \nmodel: {type(model)}\n scaler: {type(scaler)})"
    return {"predictions": prediction}
    

storage_client, bucket = initialize_client_and_bucket()
scaler = load_scaler(bucket=bucket)
model = load_model(bucket=bucket)

if __name__ == '__main__':
    print("Started predict.py ")
    app.run(host='0.0.0.0', port=os.environ["AIP_HTTP_PORT"])