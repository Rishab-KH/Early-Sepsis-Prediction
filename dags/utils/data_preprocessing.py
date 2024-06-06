# import libraries
import pandas as pd
import numpy as np
import logging
import time
import os
from pathlib import Path
import pickle
from utils import logging_setup, config


def data_preprocessing_function(df):

    columns_drop={'Unnamed: 0','SBP','DBP',
                  'EtCO2','BaseExcess', 'HCO3',
                  'pH','PaCO2','Alkalinephos', 
                  'Calcium','Magnesium', 'Phosphate',
                  'Potassium','PTT','Fibrinogen',
                  'Unit1','Unit2'}
    df["Unit"] = df["Unit1"] + df["Unit2"]

    # dropping columns based on redundancy
    df.drop(columns=columns_drop, inplace = True)
    grouped_by_patient = df.groupby('Patient_ID')

    # imputing backfill and forward fill
    df = grouped_by_patient.apply(lambda x: x.bfill().ffill())

    # dropping all the columns with null values more than 25% and patient_id
    null_columns = ['TroponinI', 
                    'Bilirubin_direct', 
                    'AST', 'Bilirubin_total', 
                    'Lactate', 'SaO2', 'FiO2',
                    'Unit', 
                    'Patient_ID']
    df = df.drop(columns=null_columns)

    # gaussian transformation
    columns_normalized = ['MAP', 'BUN', 'Creatinine', 'Glucose', 'WBC', 'Platelets' ]
    for i in columns_normalized:
        df[i] = np.log(df[i]+1)

    # onehot encoding gender column
    one_hot = pd.get_dummies(df['Gender'])
    df = df.join(one_hot)
    df = df.drop('Gender', axis=1)
    df = df.dropna()
    return df