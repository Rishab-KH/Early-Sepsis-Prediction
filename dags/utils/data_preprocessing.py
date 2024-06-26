# import libraries
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.environ["AIRFLOW_HOME"]))

# Custom imports
import utils.config as config
from utils.log_config import setup_logging
from utils.helper import load_data_from_pickle, save_data_to_pickle

# Logger setup for data preprocessing
logger = setup_logging(config.PROJECT_ROOT, "data_preprocessing.py")

def data_preprocess_pipeline(data_input, target_input, data_output):

    """
    Load, preprocess, and save the dataframe.

    Args:
        data_input (str): Path to the input pickle file containing the independent features in the dataframe.
        target_input (str): Path to the input pickle file containing the target values in the datframe.
    Return:
        Dataframe (pd.DataFrame): preprocessed dataframe containing features is returned  
    """

    try:
        # Load data from pickle file
        X = load_data_from_pickle(data_input)
        y = load_data_from_pickle(target_input)

        logger.info(f"Shape of X before preprocessing: {X.shape}")
        logger.info(f"Shape of y before preprocessing: {y.shape}")

        # Preprocess the dataframe
        columns_to_drop = ['SBP', 'DBP', 'EtCO2', 'BaseExcess', 'HCO3',
                           'pH', 'PaCO2', 'Alkalinephos', 'Calcium',
                           'Magnesium', 'Phosphate', 'Potassium', 'PTT',
                           'Fibrinogen', 'Unit1', 'Unit2']
        X["Unit"] = X["Unit1"] + X["Unit2"]

        # Drop redundant columns
        X.drop(columns=columns_to_drop, inplace=True)

        # Join X and y
        df = pd.concat([X, y], axis=1)

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
            df[col] = np.log1p(df[col])

        # One-hot encode the 'Gender' column
        df['gender'] = df['Gender'].replace({1: 'M', 0: 'F'}).apply(lambda x: x if x in ['M', 'F'] else np.nan)
        encoded_gender_col = pd.get_dummies(df["gender"])
        df = df.join(encoded_gender_col)
        df.drop(columns = ["Gender", "gender"], axis=1, inplace=True)

        # Ensure the encoded columns are integers
        df['F'] = df['F'].astype(int)
        df['M'] = df['M'].astype(int)

        # Drop remaining rows with any NaN values
        df.dropna(inplace=True)

        # Split the dataframe back into X and y
        X_preprocessed = df.drop(columns=['SepsisLabel'])
        y_preprocessed = df['SepsisLabel']

        X_preprocessed = X_preprocessed.drop(columns=X_preprocessed.columns[X_preprocessed.columns.str.contains('^Unnamed', case=False, regex=True)])

        logger.info(f"Shape of X after preprocessing: {X_preprocessed.shape}")
        logger.info(f"Shape of y after preprocessing: {y_preprocessed.shape}")

    
        # Save X_preprocesed and y_preprocessed file as a pickle file to the data_output, target_input arg mentioned in data_preprocessing func
        save_data_to_pickle(X_preprocessed, data_output)
        save_data_to_pickle(y_preprocessed, target_input)

        return X_preprocessed
        
    except KeyError as ke:
        logger.error("KeyError during preprocessing: %s", ke)
    except ValueError as ve:
        logger.error("ValueError during preprocessing: %s", ve)
    except Exception as ex:
        logger.error("An unexpected error occurred during preprocessing: %s", ex)
        raise
