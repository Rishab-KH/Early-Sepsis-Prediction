import sys
import os
import numpy as np
import pandas as pd

def read_and_concat_psv_files(folder):
    all_dataframes = []
    for filename in os.listdir(folder):
        if filename.endswith(".psv") and filename.startswith("p"):
            file_path = os.path.join(folder, filename)
            df = pd.read_csv(file_path, sep='|')
            # Extract PID from the filename
            pid = filename[2:-4]
            df['PID'] = pid
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("No PSV files found in the specified folder.")
        sys.exit(1)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df

def create_count_df(combined_df):
    # Count occurrences of SepsisLabel for each PID
    count_0 = combined_df[combined_df['SepsisLabel'] == 0].groupby('PID').size().reset_index(name='count_of_0')
    count_1 = combined_df[combined_df['SepsisLabel'] == 1].groupby('PID').size().reset_index(name='count_of_1')


