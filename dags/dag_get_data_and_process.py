from airflow import DAG
from datetime import timedelta
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator


default_args = {
    "owner": 'airflow',
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

def data_preprocessing(ti):
    df = pd.read_csv("gs://sepsis-prediction-mlops/data/modified_data/finalDataset-000000000000.csv", sep=";")

    # Fill missing values with 0s
    cols_to_fill_zero = ['Bilirubin_direct', 'TroponinI', 'Fibrinogen']
    df[cols_to_fill_zero] = df[cols_to_fill_zero].fillna(0)

    # Fill missing values with mean 
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=imputer.get_feature_names_out()) # This line is sus, removes features that does not need imputation

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    
    # Label encode columns
    if 'Gender' in df.columns:
        encoder = LabelEncoder()
        df['Gender'] = encoder.fit_transform(df['Gender'])

    X = df.drop('SepsisLabel', axis=1)
    y = df['SepsisLabel']   

    # Clip values
    X['Age_log'] = np.log1p(X['Age'].clip(lower=0))
    X['ICULOS_log'] = np.log1p(X['ICULOS'].clip(lower=0))
    X['HospAdmTime_log'] = np.log1p(-X['HospAdmTime'].clip(upper=0) + 1)

    X.drop(columns=['Age', 'ICULOS', 'HospAdmTime'], inplace=True)

    # Scale values
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Save processed data
    with open('X_data.pkl', 'wb') as file:
        pickle.dump(X_scaled, file)

    with open('y_data.pkl', 'wb') as file:
        pickle.dump(y, file)

    # Save artifacts
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    with open('imputer.pkl', 'wb') as file:
        pickle.dump(imputer, file)

    with open('encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)

with DAG(
    dag_id = "train_data_preprocess_with_gcp",
    description = "This DAG is responsible for cleaning and preprocessing the raw data into usable format",
    start_date =datetime(2024,5,15,2),
    schedule_interval = None,
    default_args=default_args,
    catchup = False
) as dag:

    task_gcs_psv_to_gcs_csv = BigQueryInsertJobOperator(
    task_id="merge_data_from_psv_to_csv",
    gcp_conn_id="my-gcp-conn",
    configuration={
        "query": {
            "query": """
            CREATE OR REPLACE EXTERNAL TABLE sepsis.dataset_temporary (
                HR FLOAT64,
                O2Sat FLOAT64,
                Temp FLOAT64,
                SBP FLOAT64,
                MAP FLOAT64,
                DBP FLOAT64,
                Resp FLOAT64,
                EtCO2 FLOAT64,
                BaseExcess FLOAT64,
                HCO3 FLOAT64,
                FiO2 FLOAT64,
                pH FLOAT64,
                PaCO2 FLOAT64,
                SaO2 FLOAT64,
                AST FLOAT64,
                BUN FLOAT64,
                Alkalinephos FLOAT64,
                Calcium FLOAT64,
                Chloride FLOAT64,
                Creatinine FLOAT64,
                Bilirubin_direct FLOAT64,
                Glucose FLOAT64,
                Lactate FLOAT64,
                Magnesium FLOAT64,
                Phosphate FLOAT64,
                Potassium FLOAT64,
                Bilirubin_total FLOAT64,
                TroponinI FLOAT64,
                Hct FLOAT64,
                Hgb FLOAT64,
                PTT FLOAT64,
                WBC FLOAT64,
                Fibrinogen FLOAT64,
                Platelets FLOAT64,
                Age FLOAT64,
                Gender INT64,
                Unit1 FLOAT64,
                Unit2 FLOAT64,
                HospAdmTime FLOAT64,
                ICULOS INT64,
                SepsisLabel INT64
            )
            OPTIONS (
            format = 'CSV',
            uris = ['gs://sepsis-prediction-mlops/data/initial_training/training_setA/training/*.psv',
                    'gs://sepsis-prediction-mlops/data/initial_training/training_setB/training_setB/*.psv'],
            skip_leading_rows = 1,
            field_delimiter="|"
            );

            EXPORT DATA OPTIONS(
            uri='gs://sepsis-prediction-mlops/data/modified_data/finalDataset-*.csv',
            format='CSV',
            overwrite=true,
            header=true,
            field_delimiter=';') AS
            SELECT * FROM sepsis.dataset_temporary LIMIT 9223372036854775807;

            DROP TABLE IF EXISTS sepsis.dataset_temporary;
            """,
            "useLegacySql": False
        }
    }
    )

    task_data_preprocessing = PythonOperator(
        task_id='data_preprocessing',
        python_callable=data_preprocessing
    )

    task_push_scaler = LocalFilesystemToGCSOperator(
       task_id="push_scaler_to_gcs",
       gcp_conn_id="my-gcp-conn",
       src="scaler.pkl",
       dst="processed_data/scaler.pkl",
       bucket="sepsis-prediction-mlops"
    )

    task_push_encoder = LocalFilesystemToGCSOperator(
       task_id="push_encoder_to_gcs",
       gcp_conn_id="my-gcp-conn",
       src="encoder.pkl",
       dst="processed_data/encoder.pkl",
       bucket="sepsis-prediction-mlops"
    )

    task_push_imputer = LocalFilesystemToGCSOperator(
       task_id="push_imputer_to_gcs",
       gcp_conn_id="my-gcp-conn",
       src="imputer.pkl",
       dst="processed_data/imputer.pkl",
       bucket="sepsis-prediction-mlops"
    )

    task_push_X_data = LocalFilesystemToGCSOperator(
       task_id="push_X_data_to_gcs",
       gcp_conn_id="my-gcp-conn",
       src="X_data.pkl",
       dst="processed_data/X_data.pkl",
       bucket="sepsis-prediction-mlops"
    )

    task_push_y_data = LocalFilesystemToGCSOperator(
       task_id="push_y_data_to_gcs",
       gcp_conn_id="my-gcp-conn",
       src="y_data.pkl",
       dst="processed_data/y_data.pkl",
       bucket="sepsis-prediction-mlops"
    )

    task_gcs_psv_to_gcs_csv >> task_data_preprocessing >> [task_push_scaler, task_push_imputer, task_push_encoder, task_push_X_data, task_push_y_data]

