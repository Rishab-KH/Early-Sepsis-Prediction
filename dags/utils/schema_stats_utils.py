#Import libraries
import pandas as pd
import numpy as np
from pathlib import Path

# Custom imports
import utils.config as config
from utils.log_config import setup_logging
from utils.data_validation import generate_and_save_schema_and_stats, validate_data

# Logger setup for schema and stats validation
DATA_DIR = config.DATA_DIR
STATS_SCHEMA_FILE = config.STATS_SCHEMA_FILE


logger = setup_logging(config.PROJECT_ROOT, "schema_stats_utils.py")

def schema_stats_gen():
    try:
        df=pd.read_csv(DATA_DIR, sep=",")
        logger.info(f"Data loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"File not found: {DATA_DIR}. Error: {e}")
        raise ValueError("Failed to Load Data for Schema and Statstics Validation. Stopping DAG execution.")

    generation_result=generate_and_save_schema_and_stats(df, STATS_SCHEMA_FILE)
    if not generation_result:
        raise ValueError("Schema and Statstics Generation failed. Stopping DAG execution.")

def schema_and_stats_validation(ti):
    try:
        print("About to run validation check")
        df=pd.read_csv(DATA_DIR, sep=",")
        logger.info(f"Data loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"File not found: {DATA_DIR}. Error: {e}")
        raise ValueError("Failed to Load Data for Schema and Statstics Validation. Stopping DAG execution.")
    validation_result, validation_message = validate_data(df)
    ti.xcom_push(key='validation_message', value=validation_message)
    if validation_result:
        return 'train_test_split'
    return 'prepare_email_content'