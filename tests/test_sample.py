### TO BE DELETED LATER

import pytest

def test_column_dropping(input_dataframe):
    df = util_data_preprocessing(input_dataframe.copy())
    columns_dropped = {'SBP', 'DBP', 'EtCO2', 'BaseExcess', 'HCO3', 'pH', 'PaCO2', 
                       'Alkalinephos', 'Calcium', 'Magnesium', 'Phosphate', 'Potassium', 
                       'PTT', 'Fibrinogen', 'Unit1', 'Unit2'}
    for column in columns_dropped:
        assert column not in df.columns

def test_null_value_handling(input_dataframe):
    df = util_data_preprocessing(input_dataframe.copy())
    assert df.isnull().sum().sum() == 0

def test_gaussian_transformation(input_dataframe):
    df = util_data_preprocessing(input_dataframe.copy())
    columns_normalized = ['MAP', 'BUN', 'Creatinine', 'Glucose', 'WBC', 'Platelets']
    for column in columns_normalized:
        assert df[column].min() > 0  # since log(0) is undefined

def test_one_hot_encoding(input_dataframe):
    df = util_data_preprocessing(input_dataframe.copy())
    assert 'M' in df.columns and 'F' in df.columns
    assert 'Gender' not in df.columns

def test_final_output_structure(input_dataframe):
    df = util_data_preprocessing(input_dataframe.copy())
    expected_columns = {'MAP', 'BUN', 'Creatinine', 'Glucose', 'WBC', 'Platelets', 'M', 'F'}
    assert set(df.columns) == expected_columns
