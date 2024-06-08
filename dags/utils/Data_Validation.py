# Import libraries
import pandas as pd
import json
import logging
from dags.utils.log_config import setup_logging
import numpy as np
import os

# Custom imports
import dags.utils.config as config
import gcsfs
import json

# setup logger

logger = setup_logging(config.PROJECT_ROOT, "data_validation.py")

def convert_to_serializable(value):
    """
    Convert a value to a JSON-serializable format.

    Args:
        value: The value to convert.

    Returns:
        The converted value.
    """

    try:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    except Exception as ex:
        logger.error(f"Error in converting JSON to serializable format {ex}")

def generate_and_save_schema_and_stats(df, schema_file):
    """
    Generate the schema and statistics from a DataFrame and save them to a JSON file.

    Args:
        df (pd.DataFrame): DataFrame from which to infer the schema and statistics.
        schema_file (str): Path to save the schema and statistics.
    """
    try: 
        schema = {col: str(df[col].dtype) for col in df.columns}
        stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    'min': convert_to_serializable(df[col].min()) if not df[col].isnull().all() else None,
                    'max': convert_to_serializable(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': convert_to_serializable(df[col].mean()) if not df[col].isnull().all() else None,
                    'median': convert_to_serializable(df[col].median()) if not df[col].isnull().all() else None,
                    'std': convert_to_serializable(df[col].std()) if not df[col].isnull().all() else None,
                    'null_count': convert_to_serializable(df[col].isnull().sum())
                }
            else:
                stats[col] = {
                    'unique_values': convert_to_serializable(df[col].unique()),
                    'null_count': convert_to_serializable(df[col].isnull().sum())
                }
        schema_and_stats = {'schema': schema, 'statistics': stats}
        with open(schema_file, 'w') as f:
            json.dump(schema_and_stats, f, indent=4)
        logger.info(f"Schema and statistics generated and saved to {schema_file}.")
        return True
    except Exception as e:
        logger.error(f"Error generating or saving schema and statistics: {e}")
        return False
        

def load_schema_and_stats(schema_file=config.STATS_SCHEMA_FILE_GCS):
    """
    Load the schema and statistics from a JSON file.

    Args:
        schema_file (str): Path to the schema and statistics file.

    Returns:
        dict: Loaded schema and statistics.
    """
    try:
        # with open(schema_file, 'r') as f:
        #     schema_and_stats = json.load(f)

        gcs_file_system = gcsfs.GCSFileSystem(project=config.GCP_PROJECT_NAME)
        gcs_json_path = config.STATS_SCHEMA_FILE_GCS
        with gcs_file_system.open(gcs_json_path) as f:
            schema_and_stats = json.load(f)
        logger.info(f"Schema and statistics loaded from {schema_file}.")
        return schema_and_stats
    except Exception as e:
        logger.error(f"Error loading schema and statistics from {schema_file}: {e}")
        raise


def validate_schema(df, schema):
    """
    Validate the schema of the DataFrame against the expected schema.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        schema (dict): Expected schema.

    Returns:
        bool: True if schema is valid, False otherwise.
    """
    for column, dtype in schema.items():
        if column not in df.columns:
            logger.error(f"Missing column: {column}")
            return False
        if str(df[column].dtype) != dtype:
            logger.error(f"Invalid type for column {column}. Expected {dtype}, got {df[column].dtype}")
            return False
    logger.info("Schema validation passed.")
    return True

def validate_statistics(df, stats):
    """
    Validate statistics of the DataFrame against expected statistics.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        stats (dict): Expected statistics.

    Returns:
        bool: True if statistics are valid, False otherwise.
    """
    try:
        for col, stat in stats.items():
            if col not in df.columns:
                logger.error(f"Missing column: {col}")
                return False
            
            if col == 'Patient_ID':
                if df[col].isnull().any():
                    logger.error("The 'patient_id' column cannot have null values.")
                    return False
                continue

            if 'min' in stat and 'max' in stat:
                if stat['min'] is not None and stat['max'] is not None:
                    if df[col].min() < stat['min']:
                        logger.warning(f"Column {col} min value anomaly: {df[col].min()} < {stat['min']}")
                    if df[col].max() > stat['max']:
                        logger.warning(f"Column {col} max value anomaly: {df[col].max()} > {stat['max']}")

            if 'mean' in stat and 'std' in stat:
                if stat['mean'] is not None and stat['std'] is not None:
                    if not df[col].isnull().all():  # Check if any non-null values exist
                        if abs(df[col].mean() - stat['mean']) > 3 * stat['std']:
                            logger.warning(f"Column {col} mean value anomaly: {df[col].mean()} != {stat['mean']}")

            if 'median' in stat and 'std' in stat:
                if stat['median'] is not None and stat['std'] is not None:
                    if not df[col].isnull().all():  # Check if any non-null values exist
                        if abs(df[col].median() - stat['median']) > 3 * stat['std']:
                            logger.warning(f"Column {col} median value anomaly: {df[col].median()} != {stat['median']}")

            if 'null_count' in stat:
                null_count = df[col].isnull().sum()
                if stat['null_count'] is not None:
                    if null_count > stat['null_count']:
                        logger.warning(f"Column {col} null value count anomaly: {null_count} > {stat['null_count']}")

            if 'unique_values' in stat:
                if stat['unique_values'] is not None:
                    unique_values = df[col].unique()
                    if set(unique_values) != set(stat['unique_values']):
                        logger.warning(f"Column {col} unique values anomaly: {unique_values} != {stat['unique_values']}")

        logger.info("Statistical validation passed.")
        return True
    except Exception as e:
        logger.error(f"Error during statistical validation: {e}")
        return False


def validate_data(df):
    """
    Validate data against stored schema and statistics.

    Args:
        file_path (str): Path to the data file.

    Returns:
        bool: True if validation passes, False if validation fails.
    """
    try:
        # Load data
        #df = pd.read_csv(file_path)
        #logger.info(f"Data loaded successfully.")

        # Load schema and statistics
        schema_and_stats = load_schema_and_stats()
        logger.info(f"schema_and_stats.json loaded successfully.")
        schema = schema_and_stats['schema']
        stats = schema_and_stats['statistics']

        # Validate schema
        if not validate_schema(df, schema):
            logger.error("Schema validation failed.")
            return False

        # Validate statistics
        if not validate_statistics(df, stats):
            logger.error("Statistical validation failed.")
            return False

        logger.info("Data validation passed.")
        return True
    except Exception as e:
        logger.error(f"Error during data validation: {e}")
        return False
