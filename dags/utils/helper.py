# Import libraries
import pickle
import os

# Custom import
import utils.config as config
from utils.log_config import setup_logging

logger = setup_logging(config.PROJECT_ROOT, "helper.py")

def load_data_from_pickle(obj_name):
    '''
    load_data_from_pickle function loads and deserializes an object from a pickle file.

    Args:
        obj_name (str): The path to the pickle file containing the serialized object. 

    Returns:
        obj (pd.DataFrame): The deserialized object that was stored in the pickle file. 

    '''
    with open(obj_name, 'rb') as file:
        obj = pickle.load(file)
    return obj

def save_data_to_pickle(obj, obj_name):
    '''
    save_data_to_pickle function saves and serializes an object from a pickle file.

    Args:
        obj (pd.DaraFrame): The pandas dataframe that needs to be serialized
        obj_name (str): The path to the pickle file containing the serialized object. 

    Returns:
        dumps serialized dataframe as pickle file to the obj_name i.e, specified path 

    '''
    with open(obj_name, 'wb') as file:
        pickle.dump(obj, file)

def clean_pickle_files(directory):
    """
    Deletes all .pkl files in the specified directory.

    Args:
        directory (str): The path to the directory where .pkl files need to be deleted.

    Returns:
        None
    """
    files = os.listdir(directory)
    for file in files:
        if file.endswith('.pkl'):
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            logger.info(f"Deleted {file_path}")

