# Import libraries
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

# Custom imports
import utils.config as config
from utils.log_config import setup_logging
from utils.helper import save_data_to_pickle

# # Logger setup for data splitting
logger = setup_logging(config.PROJECT_ROOT, "data_split.py")
DATA_DIR = config.DATA_DIR


def train_test_split():
    try:
        df = pd.read_csv(DATA_DIR, sep=",")
        train_inds, test_inds = next(GroupShuffleSplit(test_size=0.25, n_splits=2,).split(df, groups=df['Patient_ID']))
        df_train = df.iloc[train_inds] 
        df_test = df.iloc[test_inds]

        X_train = df_train.drop('SepsisLabel', axis=1)
        X_test = df_test.drop('SepsisLabel', axis=1)
        y_train = df_train['SepsisLabel']
        y_test = df_test['SepsisLabel']

        save_data_to_pickle(X_train, 'X_train.pkl')
        save_data_to_pickle(X_test, 'X_test.pkl')
        save_data_to_pickle(y_train, 'y_train.pkl')
        save_data_to_pickle(y_test, 'y_test.pkl')

    except Exception as ex:
        logger.error(f"Splitting data failed with an exception as {ex}")