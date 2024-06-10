import pytest
import pandas as pd
import numpy as np
from dags.utils.data_preprocessing import data_preprocess_pipeline 

@pytest.fixture
def input_dataframe():

    data = {
        'Patient_ID': [1, 1, 2, 2],
        'SBP': [120, 130, 140, 150],
        'DBP': [80, 85, 90, 95],
        'EtCO2': [35, 36, 37, 38],
        'BaseExcess': [2, 3, 4, 5],
        'HCO3': [22, 23, 24, 25],
        'pH': [7.4, 7.45, 7.5, 7.55],
        'PaCO2': [40, 41, 42, 43],
        'Alkalinephos': [100, 110, 120, 130],
        'Calcium': [9.0, 9.5, 10.0, 10.5],
        'Magnesium': [1.8, 1.9, 2.0, 2.1],
        'Phosphate': [3.5, 3.6, 3.7, 3.8],
        'Potassium': [4.0, 4.1, 4.2, 4.3],
        'PTT': [30, 31, 32, 33],
        'Fibrinogen': [400, 410, 420, 430],
        'Unit1': [1, 0, 1, 0],
        'Unit2': [0, 1, 0, 1],
        'TroponinI': [0.1, 0.2, np.nan, np.nan],
        'Bilirubin_direct': [0.5, np.nan, 0.7, 0.8],
        'AST': [20, 21, 22, 23],
        'Bilirubin_total': [1.0, 1.1, 1.2, 1.3],
        'Lactate': [1.5, 1.6, 1.7, 1.8],
        'SaO2': [95, 96, 97, 98],
        'FiO2': [21, 22, 23, 24],
        'MAP': [70, 75, 80, 85],
        'BUN': [10, 15, 20, 25],
        'Creatinine': [0.9, 1.0, 1.1, 1.2],
        'Glucose': [100, 110, 120, 130],
        'WBC': [5.0, 5.5, 6.0, 6.5],
        'Platelets': [200, 210, 220, 230],
        'Gender': ['M', 'F', 'M', 'F'],
    }
    target = {'SepsisLabel': [1, 0, 0, 1]}
    X = pd.DataFrame(data)
    y = pd.DataFrame(target)
    return X, y



@pytest.fixture
def save_input_files(input_dataframe):
    X, y = input_dataframe

    data_input = 'test_features.pkl'
    target_input = 'test_target.pkl'
    data_output = 'test_preprocessed_features.pkl'

    X.to_pickle(data_input)
    y.to_pickle(target_input)

    yield data_input, target_input, data_output

    # Clean up the files
    import os
    os.remove(data_input)
    os.remove(target_input)
    os.remove(data_output)

def test_column_dropping(save_input_files):
    data_input, target_input, data_output = save_input_files

    df = data_preprocess_pipeline(data_input, target_input, data_output)

    columns_dropped = {'SBP', 'DBP', 'EtCO2', 'BaseExcess', 'HCO3', 'pH', 'PaCO2', 
                       'Alkalinephos', 'Calcium', 'Magnesium', 'Phosphate', 'Potassium', 
                       'PTT', 'Fibrinogen', 'Unit1', 'Unit2'}
    for column in columns_dropped:
        assert column not in df.columns

def test_null_value_handling(save_input_files):
    data_input, target_input, data_output = save_input_files

    df = data_preprocess_pipeline(data_input, target_input, data_output)

    assert df.isnull().sum().sum() == 0

def test_gaussian_transformation(save_input_files):
    data_input, target_input, data_output = save_input_files

    df = data_preprocess_pipeline(data_input, target_input, data_output)

    columns_normalized = ['MAP', 'BUN', 'Creatinine', 'Glucose', 'WBC', 'Platelets']
    for column in columns_normalized:
        assert df[column].min() > 0  # since log(0) is undefined

def test_one_hot_encoding(save_input_files):
    data_input, target_input, data_output = save_input_files

    df = data_preprocess_pipeline(data_input, target_input, data_output)
    assert 'M' in df.columns and 'F' in df.columns
    assert 'Gender' not in df.columns

def test_final_output_structure(save_input_files):
    data_input, target_input, data_output = save_input_files

    df = data_preprocess_pipeline(data_input, target_input, data_output)

    expected_columns = {'MAP', 'BUN', 'Creatinine', 'Glucose', 'WBC', 'Platelets', 'M', 'F'}
    assert set(df.columns) == expected_columns

