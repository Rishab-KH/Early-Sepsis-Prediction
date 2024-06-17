# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Custom imports
import dags.utils.config as config
from dags.utils.log_config import setup_logging
from dags.utils.helper import load_data_from_pickle, save_data_to_pickle

logger = setup_logging(config.PROJECT_ROOT, "data_scale_utils.py")

def scale_train_data(data_pkl, scaler_pkl):
    """
    Load, scale, and save the training data and scaler.

    The function:
    - Loads the processed training data from a pickle file.
    - Scales specified columns using a StandardScaler.
    - Saves the scaled training data and the scaler object to pickle files.
    """
    try:
        X_train_processed_df = load_data_from_pickle(data_pkl)
        scaler = StandardScaler()
        columns_to_scale = ['HR', 'O2Sat', 'Temp', 'MAP', 'Resp', 'BUN', 'Chloride', 'Creatinine', 'Glucose', 'Hct', 'Hgb', 'WBC', 'Platelets']
        X_train_processed_df[columns_to_scale] = scaler.fit_transform(X_train_processed_df[columns_to_scale])
        
        save_data_to_pickle(X_train_processed_df, 'X_train_processed_scaled.pkl')
        save_data_to_pickle(scaler, scaler_pkl)
        logger.info("Training data scaled and saved successfully.")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}. Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during scaling the training data: {e}")

def scale_test_data(data_pkl, scaler_pkl):
    """
    Load, scale, and save the test data using the pre-fitted scaler.

    The function:
    - Loads the processed test data from a pickle file.
    - Loads the scaler object from a pickle file.
    - Scales specified columns using the pre-fitted StandardScaler.
    - Saves the scaled test data to a pickle file.
    """
    try:
        X_test_processed_df = load_data_from_pickle(data_pkl)
        scaler = load_data_from_pickle(scaler_pkl)
        columns_to_scale = ['HR', 'O2Sat', 'Temp', 'MAP', 'Resp', 'BUN', 'Chloride', 'Creatinine', 'Glucose', 'Hct', 'Hgb', 'WBC', 'Platelets']
        X_test_processed_df[columns_to_scale] = scaler.transform(X_test_processed_df[columns_to_scale])
        
        save_data_to_pickle(X_test_processed_df, 'X_test_processed_scaled.pkl')
        logger.info("Test data scaled and saved successfully.")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}. Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during scaling the test data: {e}")